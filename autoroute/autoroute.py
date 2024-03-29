import logging
import sys
import os
import re
import glob
import multiprocessing
from typing import Tuple, Any, List

import geopandas as gpd
import pandas as pd
import numpy as np
import psutil
import fiona
import yaml
from osgeo import gdal, osr
from shapely.geometry import box
from pyproj import Transformer

# GDAL setups
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = 'TRUE' # Set to avoid reading really large folders
os.environ["GDAL_NUM_THREADS"] = 'ALL_CPUS'
gdal.UseExceptions()

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import pyogrio
    GPD_ENGINE = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"
except ImportError:
    GPD_ENGINE = "fiona"

if gdal.GetDriverByName("Parquet") is not None:
    # If gdal parquet library is installed, use it because its faster/better compression
    GEOMETRY_SAVE_EXTENSION = "parquet"
else:
    GEOMETRY_SAVE_EXTENSION = "gpkg"


class AutoRouteHandler:
    def __init__(self, 
                 yaml: str) -> None:
        """
        yaml may be string or dictionary
        """
        self.setup(yaml)

    def run(self) -> None:
        if self.BUFFER_FILES:
            if not all({self.DEM_FOLDER, self.DEM_NAME, self.STREAM_NETWORK_FOLDER, self.LAND_USE_FOLDER, self.FLOWFILE, self.ID_COLUMN, self.FLOW_COLUMN}):
                logging.error('You\'re missing some inputs for "Buffer Files"')
                return
            dems = self.find_dems_in_extent(self.EXTENT)
        else:
            dems = [os.path.join(self.DEM_FOLDER,f) for f in os.listdir(self.DEM_FOLDER) if f.lower().endswith((".tif", ".vrt"))]
        processes = min(len(dems), os.cpu_count())
        if processes == 0:
            logging.error('No dems found in the extent. Exiting...')
            return
        
        with multiprocessing.Pool(processes=processes) as pool:
            pool.map(self.create_strm_file, dems)

            strms = glob.glob(os.path.join(self.DATA_DIR, 'stream_files', f"{self.DEM_NAME}__{self.STREAM_NAME}","*.tif"))
            if len(strms) != len(dems):
                logging.warning(f'Only {len(strms)} stream files were created out of {len(dems)} dems')
            if self.FLOWFILE:
                pool.map(self.create_row_col_id_file, strms)

    def setup(self, yaml_file) -> None:
        """
        Ensure working folder exists and set up. Each folder contains a folder based on the type of dem used and the stream network
        DATA_DIR
            dems_buffered
                FABDEM__v1-1
                    - 1.tif
                    - 2.tif
                other_dem__v1-1
            stream_files
                FABDEM__v1-1
                    - 1.tif
                    - 2.tif
            lu_buffered
                FABDEM__v1-1
                    - 1.tif
            rapid_files
                FABDEM__v1-1
                        - 1.txt
            flow_files
                FABDEM__v1-1
                        - 1.txt
            vdts
                FABDEM__v1-1
                    - 1.txt
            mifns
                FABDEM__v1-1
                    - 1.txt
        """
        # Read the yaml file and get variables out of it
        self.DATA_DIR = ""
        self.BUFFER_FILES = False
        self.DEM_FOLDER = ""
        self.DEM_NAME = ""
        self.STREAM_NETWORK_FOLDER = ""
        self.LAND_USE_FOLDER = "" 
        self.FLOWFILE = ""
        self.ID_COLUMN = ""
        self.FLOW_COLUMN = ""
        self.EXTENT = []
        self.OVERWRITE = False
        self.STREAM_NAME = ""
        self.STREAM_ID = ""
        self.BASE_FLOW_COLUMN = ""

        if isinstance(yaml_file, dict):
            for key, value in yaml_file.items():
                setattr(self, key, value)
        elif isinstance(yaml_file, str):
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    setattr(self, key, value)
        if not self.DATA_DIR:
            logging.error('No working folder provided!')
            return
        os.makedirs(self.DATA_DIR, exist_ok = True)
        #os.chdir(self.DATA_DIR)
        with open(os.path.join(self.DATA_DIR,'This is the working folder. Please delete and modify with caution.txt'), 'a') as f:
            pass

        os.makedirs(os.path.join(self.DATA_DIR,'dems_buffered'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'stream_files'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'lu_bufferd'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'rapid_files'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'flow_files'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'vdts'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'mifns'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'stream_reference_files'), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR,'tmp'), exist_ok=True)

    def buffer(self) -> None:
        pass

    def find_dems_in_extent(self, 
                            extent: list = None) -> list[str]:
        dems = self.find_files(self.DEM_FOLDER)
        if extent:
            dems = [dem for dem in dems if self.is_in_extent_gdal(dem, extent)]
        return dems

    def find_number_of_dems_in_extent(self) -> int:
        dems = self.find_dems_in_extent()
        return len(dems)

    def find_files(self, directory: str, type: str = '*.tif') -> list[str]:
        tif_files = []
        for root, _, _ in os.walk(directory):
            tif_files.extend(glob.glob(os.path.join(root, type)))
        return tif_files

    def is_in_extent_gdal(self, 
                          dem: str,
                          extent: list = None) -> bool:
        if extent is None:
            return True
        ds = self.open_w_gdal(dem)
        if not ds: return False

        minx, miny, maxx,maxy = self.get_ds_extent(ds)
        ds = None

        minx1, miny1, maxx1, maxy1 = extent 
        if (minx1 <= maxx and maxx1 >= minx and miny1 <= maxy and maxy1 >= miny):
            return True
        return False
    
    def buffer_dem(self, dem: str) -> None:
        ds = self.open_w_gdal(dem)
        if not ds: return
        projection = ds.GetProjection()
        no_data_value = ds.GetRasterBand(1).GetNoDataValue()

        try:
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            if srs.IsProjected():
                units = srs.GetLinearUnitsName()
            else:
                units = srs.GetAngularUnitsName()
            if 'degree' in units:
                buffer_distance = 0.1 # Buffer by 0.1 degrees
            elif 'meter' in units and not 'k' in units:
                buffer_distance = 10_000 # Buffer by 10 km
            else:
                raise NotImplementedError(f'Unsupported units: {units}')
        except Exception as e:
            logging.error(f'Error buffering dem: \n{e}')
        minx, miny, maxx,maxy = self.get_ds_extent(ds)
        ds = None
        minx -= buffer_distance
        miny -= buffer_distance
        maxx +=  buffer_distance
        maxy +=  buffer_distance

        if buffer_distance == 0.1:
            minx = max(minx, -180)
            miny = max(miny, -90)
            maxx = min(maxx, 180)
            maxy = min(maxy, 90)
        # TODO figure out if gdal has errors for out of bounds projected units

        os.makedirs(os.path.join(self.DATA_DIR, 'dems_buffered',self.DEM_NAME), exist_ok=True)
        buffered_dem = os.path.join(self.DATA_DIR, 'dems_buffered', self.DEM_NAME, f"{str(round(minx, 3)).replace('.','_')}__{str(round(miny, 3)).replace('.','_')}__{str(round(maxx, 3)).replace('.','_')}__{str(round(maxy, 3)).replace('.','_')}.vrt")
        if self.OVERWRITE and os.path.exists(buffered_dem):
            logging.info(f"{buffered_dem} already exists. Skipping...")
            return

        dems = self.find_dems_in_extent(extent=(minx, miny, maxx, maxy))
        if not dems:
            logging.warning(f'Somehow no dems found in this extent: {(minx, miny, maxx, maxy)}')
            return
        
        # vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
        # vrt_dataset = gdal.BuildVRT('', dems, options=vrt_options)
        # warp_options = gdal.WarpOptions(
        #     format='GTiff',
        #     dstSRS=projection,
        #     dstNodata=no_data_value,
        #     outputBounds=(minx, miny, maxx, maxy),
        #     outputType=gdal.GDT_Float32,  
        #     multithread=True, 
        #     creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=3","BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
        #     )    
        
        try:
            #gdal.Warp(buffered_dem, vrt_dataset, options=warp_options)
            vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear',
                                               outputSRS=projection,
                                               srcNodata=no_data_value,
                                               outputBounds=(minx, miny, maxx, maxy))
            gdal.BuildVRT(buffered_dem, dems, options=vrt_options)

        except RuntimeError as e:
            try:
                disk, required = re.findall(r"(\d+)", str(e))
                raise MemoryError(f"Need {self._sizeof_fmt(int(required))}; {self._sizeof_fmt(int(disk))} of space on this machine")
            except:
                logging.error(f'Error buffering dem: \n{e}')

        # Clean up the VRT dataset
        vrt_dataset = None
        logging.info(f'Suceeded buffer: {buffered_dem}')
        
    def _sizeof_fmt(self, num:int) -> str:
        """
        Take in an int number of bytes, outputs a string that is human readable
        """
        for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
            if abs(num) < 1024.0:
                return f"{num:3.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} YB"

    def open_w_gdal(self, path: str) -> gdal.Dataset:
        ds = gdal.OpenEx(path)
        if not ds:
            logging.warning(f'Could not open {path}. Is this a valid file?')
        return ds
    
    def get_ds_extent(self, ds: gdal.Dataset) -> Tuple[float, float, float, float]:
        geo = ds.GetGeoTransform()
        minx = geo[0]
        maxy = geo[3]
        maxx = minx + geo[1] * ds.RasterXSize
        miny = maxy + geo[5] * ds.RasterYSize
        return minx, miny, maxx, maxy
    
    def gpd_read(self, 
                 path: str,
                 bbox: box = None,
                 columns = None,) -> gpd.GeoDataFrame:
        """
        Reads .parquet, .shp, .gpkg, and potentially others and returns in a way we expect.
        """
        global GPD_ENGINE
        if path.endswith((".parquet", ".geoparquet")):
            if columns:
                df = gpd.read_parquet(path, columns=columns)
            else:
                df = gpd.read_parquet(path)
            if bbox:
                df = df[df.geometry.intersects(bbox)]
        elif path.endswith((".shp", ".gpkg")):
            if bbox:
                df = gpd.read_file(path, bbox=bbox, engine=GPD_ENGINE)
            else:
                df = gpd.read_file(path, engine=GPD_ENGINE)
            if columns:
                df = df[columns]
        else:
            raise NotImplementedError(f"Unsupported file type: {path}")
        return df
            
    def create_strm_file(self, dem: str) -> None:
        global GEOMETRY_SAVE_EXTENSION
        ds = self.open_w_gdal(dem)
        if not ds: return
        projection = ds.GetProjection()
        ds_epsg = int(ds.GetSpatialRef().GetAttrValue('AUTHORITY', 1))

        strm = os.path.join(self.DATA_DIR, 'stream_files', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(strm, exist_ok=True)
        strm = os.path.join(strm, f"{os.path.basename(dem).split('.')[0]}__strm.tif")
        if not self.OVERWRITE and not os.path.exists(strm):
            logging.info(f"{strm} already exists. Skipping...")
            return
        
        minx, miny, maxx, maxy = self.get_ds_extent(ds)

        filenames = [os.path.join(self.STREAM_NETWORK_FOLDER,f) for f in os.listdir(self.STREAM_NETWORK_FOLDER) if f.endswith(('.shp', '.gpkg', '.parquet', '.geoparquet'))]
        if not filenames:
            logging.warning(f"No stream files found in {self.STREAM_NETWORK_FOLDER}")
            return
        tmp_streams = os.path.join(self.DATA_DIR, 'tmp', f'temp.{GEOMETRY_SAVE_EXTENSION}')
        try:
            dfs = []
            for f in filenames:
                with fiona.open(f, 'r') as src:
                    crs = src.crs
                if ds_epsg != crs.to_epsg():
                    transformer = Transformer.from_crs(f"EPSG:{ds_epsg}",crs.to_string(), always_xy=True) 
                    minx2, miny2 = transformer.transform(minx, miny)
                    maxx2, maxy2 =  transformer.transform(maxx, maxy)
                    bbox = box(minx2, miny2, maxx2, maxy2)
                else:
                    bbox = box(minx, miny, maxx, maxy)

                try:
                    df = self.gpd_read(f, columns=[self.STREAM_ID, 'geometry'], bbox=bbox)
                except NotImplementedError:
                    logging.warning(f"Skipping unsupported file: {f}")
                    continue
                if not df.empty:
                    dfs.append(df.to_crs(ds_epsg))
                else:
                    logging.warning(f"No streams were found in {f}")
            df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
            if GEOMETRY_SAVE_EXTENSION == "parquet":
                df.to_parquet(tmp_streams)
            else:
                df.to_file(tmp_streams)
        except KeyError as e:
            logging.error(f"{self.STREAM_ID} not found in the stream files here: {self.STREAM_NETWORK_FOLDER}")

        options = gdal.RasterizeOptions(attribute=self.STREAM_ID,
                              outputType=gdal.GDT_UInt64, # Assume no negative IDs
                              format='GTiff',
                              outputSRS=projection,
                              creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
                              outputBounds=(minx, miny, maxx, maxy),
                              noData=0,
                              width=ds.RasterXSize,
                              height=ds.RasterYSize,
        )
        gdal.Rasterize(strm, tmp_streams, options=options)
        logging.debug(f"Rasterizing {dem} using {self.STREAM_NETWORK_FOLDER} to {strm}")
        ds = None

    def create_row_col_id_file(self, strm: str) -> None:
        row_col_file = os.path.join(self.DATA_DIR, 'rapid_files', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(row_col_file, exist_ok=True)
        row_col_file = os.path.join(row_col_file, f"{os.path.basename(strm).split('.')[0]}__row_col_id.txt")
        if not self.OVERWRITE and not os.path.exists(row_col_file):
            logging.info(f"{row_col_file} already exists. Skipping...")
            return

        if self.FLOWFILE.endswith(('.csv', '.txt')):
            df = (
                pd.read_csv(self.FLOWFILE, sep=',')
                .drop_duplicates(self.ID_COLUMN if self.ID_COLUMN else None) 
                .dropna()
            )
            if df.empty:
                logging.error(f"No data in the flow file: {self.FLOWFILE}")
                return
            if not self.ID_COLUMN:
                self.ID_COLUMN =df.columns[0]
            if self.ID_COLUMN not in df:
                raise ValueError(f"The id field you've entered is not in the file\nids entered: {self.ID_COLUMN}, columns found: {list(df)}")
            if not self.FLOW_COLUMN:
                cols = df.columns
            else:
                if self.FLOW_COLUMN not in df.columns:
                    raise ValueError(f"The flow field you've entered is not in the file\nflows entered: {self.FLOW_COLUMN}, columns found: {list(df)}")
                cols = [self.ID_COLUMN, self.FLOW_COLUMN]
                if self.BASE_FLOW_COLUMN:
                    if self.BASE_FLOW_COLUMN not in df.columns:
                        raise ValueError(f"The flow field you've entered is not in the file\nflows entered: {self.FLOW_COLUMN}, columns found: {list(df)}")
                    cols.append(self.BASE_FLOW_COLUMN)
                    
            df = df[cols]
        else:
            raise NotImplementedError(f"Unsupported file type: {self.FLOWFILE}")
                
        # Now look through the Raster to find the appropriate Information and print to the FlowFile
        ds = gdal.Open(strm)
        data_array = ds.ReadAsArray()
        ds = None

        indices = np.where(data_array > 0)
        values = data_array[indices]
        matches = df[df[self.ID_COLUMN].isin(values)].shape[0]

        if matches != df.shape[0]:
            logging.warning(f"{matches} id(s) out of {df.shape[0]} from your input file are present in the stream raster...")
        
        (
            pd.DataFrame({'ROW': indices[0], 'COL': indices[1], self.ID_COLUMN: values})
            .merge(df, on=self.ID_COLUMN, how='left')
            .fillna(0)
            .to_csv(row_col_file, sep=" ", index=False)
        )
        logging.info("Finished stream file")

    def list_to_sublists(self, alist: List[Any], n: int) -> List[List[Any]]:
        return [alist[x:x+n] for x in range(0, len(alist), n)]
        
    
