import logging
import sys
import os
import math
import re
import glob
import multiprocessing
from typing import Tuple, Any, List, Set
import tqdm
import subprocess
import asyncio

import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
import yaml
from osgeo import gdal, osr
from shapely.geometry import box
from pyproj import Transformer

# GDAL setups
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = 'TRUE' # Set to avoid reading really large folders
os.environ["GDAL_NUM_THREADS"] = 'ALL_CPUS'
gdal.UseExceptions()

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
                 yaml: str = None) -> None:
        """
        yaml may be string or dictionary
        """
        if yaml:
            self.setup(yaml)

    def run(self) -> None:
        dems = []
        strms = []
        lus = []
        sim_flow_files = []
        flood_files = []
        if self.DEM_FOLDER:
            dems = {os.path.join(self.DEM_FOLDER,f) for f in os.listdir(self.DEM_FOLDER) if f.lower().endswith((".tif", ".vrt"))}
        else:
            if self.BUFFER_FILES:
                dems = {f for f in glob.glob(os.path.join(self.DATA_DIR, 'dems', self.DEM_NAME, "*_buff.*")) if f.lower().endswith((".tif", ".vrt"))}
            elif self.CROP:
                dems = {f for f in glob.glob(os.path.join(self.DATA_DIR, 'dems', self.DEM_NAME, "*_crop.*")) if f.lower().endswith((".tif", ".vrt"))}
        if self.EXTENT:
            dems = {dem for dem in dems if self.is_in_extent(dem, self.EXTENT)}
        processes = max(min(len(dems), os.cpu_count()), 1)
        mifns = None
        if processes > 0:
            with multiprocessing.Pool(processes=processes) as pool:
                if self.CROP and self.EXTENT:
                    dems = pool.map(self.crop, dems)

                if self.STREAM_NETWORK_FOLDER:
                    strms = pool.map(self.create_strm_file, dems)

                if self.LAND_USE_FOLDER:
                    lus = pool.map(self.create_land_use, dems)

                # strms = glob.glob(os.path.join(self.DATA_DIR, 'stream_files', f"{self.DEM_NAME}__{self.STREAM_NAME}","*.tif"))
                if not self.DEM_FOLDER and self.EXTENT:
                    strms = {strm for strm in strms if self.is_in_extent(strm, self.EXTENT)}
                if strms and self.SIMULATION_FLOWFILE:
                    sim_flow_files = pool.map(self.create_row_col_id_file, strms)

                if self.FLOOD_FLOWFILE:
                    flood_files = pool.map(self.create_flood_flowfile, strms)
                
                pairs = self._zip_files(dems, strms, lus, sim_flow_files, flood_files)
                if self.AUTOROUTE or self.FLOODSPREADER:
                    mifns = pool.starmap(self.create_mifn_file, pairs)

        if mifns and (self.AUTOROUTE or self.FLOODSPREADER):
            processes = min(len(mifns), os.cpu_count())
            with multiprocessing.Pool(processes=processes) as pool:
                if self.AUTOROUTE:
                    results = tqdm.tqdm(pool.imap_unordered(self.run_autoroute, mifns))
                    for res in results:
                        print(results)


        
    def setup(self, yaml_file) -> None:
        """
        Ensure working folder exists and set up. Each folder contains a folder based on the type of dem used and the stream network
        DATA_DIR
            dems
                FABDEM__v1-1
                    - 1.tif
                    - 2.tif
                    - 1_crop.tif
                other_dem__v1-1
            stream_files
                FABDEM__v1-1
                    - 1.tif
                    - 2.tif
            land_use
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
            meta_files
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
        self.SIMULATION_FLOWFILE = ""
        self.FLOOD_FLOWFILE = ""
        self.ID_COLUMN = ""
        self.FLOW_COLUMN = ""
        self.EXTENT = None
        self.CROP = False
        self.OVERWRITE = False
        self.STREAM_NAME = ""
        self.STREAM_ID = ""
        self.BASE_FLOW_COLUMN = ""
        self.LAND_USE_NAME = ""
        self.MANNINGS_TABLE = ""

        self.AUTOROUTE = ""
        self.FLOODSPREADER = ""
        self.AUTOROUTE_CONDA_ENV = ""

        self.RAPID_Subtract_BaseFlow = False
        self.VDT = ""
        self.num_iterations = 15
        self.convert_cfs_to_cms = False
        self.x_distance = 1000
        self.q_limit = 1.1
        self.direction_distance = 1
        self.slope_distance = 1
        self.weight_angles = 0
        self.use_prev_d_4_xs = 1
        self.adjust_flow = 1
        self.Str_Limit_Val = 0
        self.UP_Str_Limit_Val = float('inf')
        self.row_start = 0
        self.row_end = float('inf')
        self.degree_manip = 0.0
        self.degree_interval = 0.0
        self.man_n = 0.01
        self.low_spot_distance = 2
        self.low_spot_is_meters = False
        self.low_spot_use_box = False
        self.box_size = 1
        self.find_flat = False
        self.low_spot_find_flat_cutoff = float('inf')
        self.ar_bathy = False
        self.bathy_alpha = 0.001
        self.bathy_method = ''
        self.bathy_x_max_depth = 0.2
        self.bathy_y_shallow = 0.2
        self.da_flow_param = ''
        self.omit_outliers = ''
        self.wse_search_dist = 10
        self.wse_threshold = 0.25
        self.wse_remove_three = False
        self.specify_depth = 0
        self.twd_factor = 1.5
        self.only_streams = False
        self.use_ar_top_widths = False
        self.flood_local = False
        self.DEPTH_MAP = ''
        self.FLOOD_MAP = ''
        self.VELOCITY_MAP = ''
        self.WSE_MAP = ''
        self.fs_bathy = False
        self.fs_bathy_smooth_method = ''
        self.bathy_twd_factor = 1

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
        if not os.path.isabs(self.DATA_DIR):
            self.DATA_DIR = os.path.abspath(self.DATA_DIR)
        os.makedirs(self.DATA_DIR, exist_ok = True)
        #os.chdir(self.DATA_DIR)
        with open(os.path.join(self.DATA_DIR,'This is the working folder. Please delete and modify with caution.txt'), 'a') as f:
            pass

        os.makedirs(os.path.join(self.DATA_DIR,'dems'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'stream_files'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'land_use'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'rapid_files'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'flow_files'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'vdts'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'mifns'), exist_ok = True)
        os.makedirs(os.path.join(self.DATA_DIR,'stream_reference_files'), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR,'tmp'), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, 'meta_files'), exist_ok=True)

    def buffer(self) -> None:
        pass

    def find_dems_in_extent(self, 
                            extent: list = None) -> list[str]:
        dems = self.find_files(self.DEM_FOLDER)
        if extent:
            dems = [dem for dem in dems if self.is_in_extent(dem, extent)]
        return dems

    def find_number_of_dems_in_extent(self) -> int:
        dems = self.find_dems_in_extent()
        return len(dems)

    def find_files(self, directory: str, type: str = '*.tif') -> list[str]:
        tif_files = []
        for root, _, _ in os.walk(directory):
            tif_files.extend(glob.glob(os.path.join(root, type)))
        return tif_files

    def is_in_extent(self, 
                          ds: str | gdal.Dataset,
                          extent: list = None) -> bool:
        if extent is None:
            return True
        if isinstance(ds, str):
            ds = self.open_w_gdal(ds)
        if not ds: return False

        minx, miny, maxx,maxy = self.get_ds_extent(ds)
        ds = None
        return self._isin(minx, miny, maxx, maxy, extent)
    
    def _isin(self, 
              minx: float, 
              miny: float, 
              maxx: float, 
              maxy: float, 
              extent: list) -> bool:
        minx1, miny1, maxx1, maxy1 = extent 
        if (minx1 <= maxx and maxx1 >= minx and miny1 <= maxy and maxy1 >= miny):
            return True
        return False
    
    def buffer_dem(self, dem: str) -> str:
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

        os.makedirs(os.path.join(self.DATA_DIR, 'dems',self.DEM_NAME), exist_ok=True)
        buffered_dem = os.path.join(self.DATA_DIR, 'dems', self.DEM_NAME, f"{str(round(minx, 3)).replace('.','_')}__{str(round(miny, 3)).replace('.','_')}__{str(round(maxx, 3)).replace('.','_')}__{str(round(maxy, 3)).replace('.','_')}_buff.vrt")
        if self.OVERWRITE and os.path.exists(buffered_dem):
            logging.info(f"{buffered_dem} already exists. Skipping...")
            return buffered_dem

        dems = self.find_dems_in_extent(extent=(minx, miny, maxx, maxy))
        if not dems:
            logging.warning(f'Somehow no dems found in this extent: {(minx, miny, maxx, maxy)}')
            return
        
        try:
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
            
    def create_strm_file(self, 
                         dem: str, ) -> str:
        global GEOMETRY_SAVE_EXTENSION
        ds = self.open_w_gdal(dem)
        if not ds: return
        projection = ds.GetProjection()
        ds_epsg = self.get_epsg(ds)

        strm = os.path.join(self.DATA_DIR, 'stream_files', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(strm, exist_ok=True)
        strm = os.path.join(strm, f"{os.path.basename(dem).split('.')[0]}__strm.tif")
        if not self.OVERWRITE and os.path.exists(strm):
            logging.info(f"{strm} already exists. Skipping...")
            return strm
        
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
                except ValueError as e:
                    logging.error(f"Cannot find {self.STREAM_ID} in {f}")
                    raise exception.Invalid
                
                if not df.empty:
                    dfs.append(df.to_crs(ds_epsg))
                else:
                    logging.warning(f"No streams were found in {f}")
            if dfs:
                df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
            else:
                logging.warning(f"No streams were found in the extent of {dem}")
                return
            if GEOMETRY_SAVE_EXTENSION == "parquet":
                df.to_parquet(tmp_streams)
            else:
                df.to_file(tmp_streams)
        except KeyError as e:
            logging.error(f"{self.STREAM_ID} not found in the stream files here: {self.STREAM_NETWORK_FOLDER}")

        options = gdal.RasterizeOptions(attribute=self.STREAM_ID,
                              outputType=gdal.GDT_UInt32, # Assume no negative IDs. DO NOT SET TO UINT64
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
        return strm

    def create_row_col_id_file(self, strm: str) -> str:
        row_col_file = os.path.join(self.DATA_DIR, 'rapid_files', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(row_col_file, exist_ok=True)
        row_col_file = os.path.join(row_col_file, f"{os.path.basename(strm).split('.')[0]}__row_col_id.txt")
        if not self.OVERWRITE and os.path.exists(row_col_file):
            logging.info(f"{row_col_file} already exists. Skipping...")
            return row_col_file

        if self.SIMULATION_FLOWFILE.endswith(('.csv', '.txt')):
            df = (
                pd.read_csv(self.SIMULATION_FLOWFILE, sep=',')
                .drop_duplicates(self.ID_COLUMN if self.ID_COLUMN else None) 
                .dropna()
            )
            if df.empty:
                logging.error(f"No data in the flow file: {self.SIMULATION_FLOWFILE}")
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
                        raise ValueError(f"The baseflow field you've entered is not in the file\nflows entered: {self.FLOW_COLUMN}, columns found: {list(df)}")
                    cols.append(self.BASE_FLOW_COLUMN)
                    
            df = df[cols]
        else:
            raise NotImplementedError(f"Unsupported file type: {self.SIMULATION_FLOWFILE}")
                
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
        return row_col_file

    def create_flood_flowfile(self, strm: str) -> str:
        flowfile = os.path.join(self.DATA_DIR, 'flow_files', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(flowfile, exist_ok=True)
        flowfile = os.path.join(flowfile, f"{os.path.basename(strm).split('.')[0]}__flow.txt")
        if not self.OVERWRITE and os.path.exists(flowfile):
            logging.info(f"{flowfile} already exists. Skipping...")
            return flowfile

        if self.FLOOD_FLOWFILE.endswith(('.csv', '.txt')):
            df = (
                pd.read_csv(self.FLOOD_FLOWFILE, sep=',')
            )
            id_col = df.columns[0]
            df = (
                df.drop_duplicates(id_col) 
                .dropna()
            )
            if df.empty:
                logging.error(f"No data in the flow file: {self.FLOOD_FLOWFILE}")
                return
        else:
            raise NotImplementedError(f"Unsupported file type: {self.FLOOD_FLOWFILE}")
                
        # Now look through the Raster to find the appropriate Information and print to the FlowFile
        ds = gdal.Open(strm)
        data_array = ds.ReadAsArray()
        ds = None

        ids = np.unique(data_array)[1:]
        
        (
            df[df[id_col].isin(ids)]
            .to_csv(flowfile,index=False)
        )
        logging.info("Finished flow file")
        return flowfile

    def create_land_use(self, dem: str) -> str:
        global GEOMETRY_SAVE_EXTENSION
        lu_file = os.path.join(self.DATA_DIR, 'land_use', f"{self.DEM_NAME}__{self.LAND_USE_NAME}")
        os.makedirs(lu_file, exist_ok=True)
        lu_file = os.path.join(lu_file, f"{os.path.basename(dem).split('.')[0]}__lu.vrt")
        if not self.OVERWRITE and os.path.exists(lu_file):
            logging.info(f"{lu_file} already exists. Skipping...")
            return lu_file
        dem_ds = self.open_w_gdal(dem)
        if not dem_ds: 
            logging.warning(f'Could not open {dem}. Skipping...')
            return
        
        dem_epsg = self.get_epsg(dem_ds)
        projection = dem_ds.GetProjection()
        dem_spatial_ref = dem_ds.GetSpatialRef()

        minx, miny, maxx, maxy = self.get_ds_extent(dem_ds)
        x_res = abs(dem_ds.GetGeoTransform()[1])
        y_res = abs(dem_ds.GetGeoTransform()[5])
        dem_ds = None

        filenames = [os.path.join(self.LAND_USE_FOLDER,f) for f in os.listdir(self.LAND_USE_FOLDER) if f.endswith(('.tif'))]
        if not self.check_has_same_projection(filenames):
            logging.error("Some land use files have different projections. Exiting...")
            raise NotImplementedError("Different projections")
        if not filenames:
            logging.warning(f"No land use files found in {self.LAND_USE_FOLDER}")
            return
        
        # Loop over every file in the land use folder and check if it intersects with the dem
        lu_epsg = self.get_epsg(self.open_w_gdal(filenames[0]))
        if dem_epsg != lu_epsg:
            out_spatial_ref = osr.SpatialReference()
            out_spatial_ref.ImportFromEPSG(lu_epsg)
            coordTrans = osr.CoordinateTransformation(dem_spatial_ref, out_spatial_ref)
        files_to_use = []
        for f in filenames:
            ds = self.open_w_gdal(f)
            if not ds: 
                logging.warning(f'Could not open {f}. Skipping...')
                continue
            minx2, miny2, maxx2, maxy2 = self.get_ds_extent(ds)
            if dem_epsg != lu_epsg:
                minx2, miny2, _ = coordTrans.TransformPoint(minx, miny)
                maxx2, maxy2, _ = coordTrans.TransformPoint(maxx, maxy)
                
            if self.is_in_extent(ds, (minx2, miny2, maxx2, maxy2)):
                files_to_use.append(f)
            ds = None
        
        if not files_to_use:
            logging.warning(f"No land use files found in the extent of {dem}")
            return
        
        if dem_epsg != lu_epsg:
            options = gdal.WarpOptions(
            format='GTiff',
            dstSRS=projection,
            outputBounds=(minx, miny, maxx, maxy),
            outputType=gdal.GDT_Byte,  
            multithread=True, 
            xRes=x_res,
            yRes=y_res,
            creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=2", "NUM_THREADS=ALL_CPUS"]
            )    
            lu_file = lu_file.replace('.vrt', '.tif')
            gdal.Warp(lu_file, files_to_use, options=options) 
        else:
            vrt_options = gdal.BuildVRTOptions(outputSRS=projection,
                                           outputBounds=(minx, miny, maxx, maxy),
                                           resampleAlg='bilinear',
                                           xRes=x_res,
                                           yRes=y_res,
                                           )
            gdal.BuildVRT(lu_file, files_to_use, options=vrt_options)
        return lu_file

    def check_has_same_projection(self, files: List[str]) -> bool:
        if not files: return True
        try:
            ds_code = self.get_epsg(self.open_w_gdal(files[0]))
            for f in files[1:]:
                if ds_code != self.get_epsg(self.open_w_gdal(f)):
                    return False
        except:
            logging.warning(f"Problems with a file")
            return False
        return True

    def get_epsg(self, ds: gdal.Dataset) -> int:
        dem_spatial_ref = ds.GetSpatialRef()
        dem_sr = osr.SpatialReference(str(dem_spatial_ref)) # load projection
        res = dem_sr.AutoIdentifyEPSG() # identify EPSG
        return int(dem_sr.GetAuthorityCode(None)) # get EPSG code

    def list_to_sublists(self, alist: List[Any], n: int) -> List[List[Any]]:
        return [alist[x:x+n] for x in range(0, len(alist), n)]
    
    def crop(self, dem: str) -> str:
        os.makedirs(os.path.join(self.DATA_DIR, 'dems',self.DEM_NAME), exist_ok=True)
        cropped_dem = os.path.join(self.DATA_DIR, 'dems', self.DEM_NAME, f"{str(round(self.EXTENT[0], 3)).replace('.','_')}__{str(round(self.EXTENT[1], 3)).replace('.','_')}__{str(round(self.EXTENT[2], 3)).replace('.','_')}__{str(round(self.EXTENT[3], 3)).replace('.','_')}_crop.vrt")
        if not self.OVERWRITE and os.path.exists(cropped_dem):
            logging.info(f"{cropped_dem} already exists. Skipping...")
            return
        ds = self.open_w_gdal(dem)
        if not ds:
            logging.warning(f'Could not open {dem}. Skipping...')
            return
        projection = ds.GetProjection()
        no_data_value = ds.GetRasterBand(1).GetNoDataValue()
        minx, miny, maxx, maxy = self.get_ds_extent(ds)
        ds = None

        # Crop the DEM to the min of the extent and the DEM itself
        minx = max(minx, self.EXTENT[0])
        miny = max(miny, self.EXTENT[1])
        maxx = min(maxx, self.EXTENT[2])
        maxy = min(maxy, self.EXTENT[3])

        vrt_options = gdal.BuildVRTOptions(resampleAlg='lanczos',
                                        outputSRS=projection,
                                        srcNodata=no_data_value,
                                        outputBounds=(minx, miny, maxx, maxy))
        gdal.BuildVRT(cropped_dem, dem, options=vrt_options)
        return cropped_dem
            
    def create_mifn_file(self, 
                         dem: str, 
                         strm: str, 
                         lu: str, 
                         rapid_flow_file: str, 
                         flowfile: str) -> str:
        # Format path strings
        dem = self._format_files(dem)
        mifn = os.path.join(self.DATA_DIR, 'mifns', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(mifn, exist_ok=True)
        mifn = os.path.join(mifn, f"{os.path.basename(dem).split('.')[0]}__mifn.txt")
        
        if not self.OVERWRITE and os.path.exists(mifn):
            return mifn
        
        meta_file = os.path.join(self.DATA_DIR, 'meta_files', f"{self.DEM_NAME}__{self.STREAM_NAME}")
        os.makedirs(meta_file, exist_ok=True)
        meta_file = os.path.join(meta_file, f"{os.path.basename(dem).split('.')[0]}__meta.txt")

        with open(mifn, 'w') as f:
            self._warn_DNE('DEM', dem)
            self._check_type('DEM', dem, ['.tif','.vrt'])
            self._write(f,'DEM_File',dem)

            f.write('\n')
            self._write(f,'# AutoRoute Inputs')
            f.write('\n')

            if strm:
                strm = self._format_files(strm)
                self._warn_DNE('Stream File', strm)
                self._check_type('Stream Fil', strm, ['.tif', '.vrt'])
                self._write(f,'Stream_File',strm)

            # Open DEM to see if projected or geographic units
            ds = gdal.Open(dem)
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            nrows = ds.RasterYSize
            ds = None
            if srs.IsProjected():
                units = srs.GetLinearUnitsName().lower()
                if "meter" in units:
                    if "k" in units:
                        self._write(f,'Spatial_Units',"km")
                    else:
                        self._write(f,'Spatial_Units',"m")
                else:
                    raise ValueError(f"Unsupported units: {units}")
            else:
                units = srs.GetAngularUnitsName().lower()
                if "degree" in units:
                    self._write(f,'Spatial_Units',"deg")
                else:
                    raise ValueError(f"Unsupported units: {units}")

            if rapid_flow_file:
                rapid_flow_file = self._format_files(rapid_flow_file)
                self._warn_DNE('Flow File', rapid_flow_file)
                self._check_type('Flow File', rapid_flow_file, ['.txt'])
                self._write(f,'Flow_RAPIDFile',rapid_flow_file)

                self._write(f,'RowCol_From_RAPIDFile')
                if not self.ID_COLUMN:
                    logging.warning('Flow ID is not specified!!')
                else:
                    self._write(f,'RAPID_Flow_ID', self.ID_COLUMN)
                if not self.FLOW_COLUMN:
                    logging.warning('Flow Params are not specified!!')
                else:
                    if not isinstance(self.FLOW_COLUMN, list):
                        self.FLOW_COLUMN = [self.FLOW_COLUMN]
                    self._write(f,'RAPID_Flow_Param', " ".join(self.FLOW_COLUMN))
                if self.RAPID_Subtract_BaseFlow:
                    if not self.BASE_FLOW_COLUMN:
                        logging.warning('Base Flow Parameter is not specified, not subtracting baseflow')
                    else:
                        self._write(f,'RAPID_BaseFlow_Param',self.BASE_FLOW_COLUMN)
                        self._write(f,'RAPID_Subtract_BaseFlow')

            if self.VDT:
                vdt = os.path.join(self.VDT, f"{os.path.basename(dem).split('.')[0]}__vdt.txt")
            else:
                vdt = os.path.join(self.DATA_DIR, 'vdts', f"{self.DEM_NAME}__{self.STREAM_NAME}", f"{os.path.basename(dem).split('.')[0]}__vdt.txt")
            vdt = self._format_files(vdt)
            self._check_type('VDT', vdt, ['.txt'])
            self._write(f,'Print_VDT_Database',vdt)
            self._write(f,'Print_VDT_Database_NumIterations',self.num_iterations)

            meta_file = self._format_files(meta_file)
            self._write(f,'Meta_File',meta_file)

            if self.convert_cfs_to_cms: self._write(f,'CONVERT_Q_CFS_TO_CMS')

            self._write(f,'X_Section_Dist',self.x_distance)
            self._write(f,'Q_Limit',self.q_limit)
            self._write(f,'Gen_Dir_Dist',self.direction_distance)
            self._write(f,'Gen_Slope_Dist',self.slope_distance)
            if self.weight_angles:
                self._write(f,'Weight_Angles',self.weight_angles)
            self._write(f,'Use_Prev_D_4_XS',self.use_prev_d_4_xs)
            self._write(f,'ADJUST_FLOW_BY_FRACTION',self.adjust_flow)
            if self.Str_Limit_Val: self._write(f,'Str_Limit_Val',self.Str_Limit_Val)
            if not math.isinf(self.UP_Str_Limit_Val): self._write(f,'UP_Str_Limit_Val',self.UP_Str_Limit_Val)
            if self.row_start: self._write(f,'Layer_Row_Start',self.row_start)
            if self.row_end < nrows: self._write(f,'Layer_Row_End',self.row_end)

            if self.degree_manip > 0 and self.degree_interval > 0:
                self._write(f,'Degree_Manip',self.degree_manip)
                self._write(f,'Degree_Interval',self.degree_interval)

            if lu:
                lu_raster = self._format_files(lu)
                self._warn_DNE('Land Use', lu_raster)
                self._check_type('Land Use', lu_raster, ['.tif', '.vrt'])
                self._write(f,'LU_Raster_SameRes',lu_raster)
                if not self.MANNINGS_TABLE:
                    logging.warning('No mannings table for the Land Use raster!')
                else:
                    mannings_table = self._format_files(self.MANNINGS_TABLE)
                    self._write(f,'LU_Manning_n',mannings_table)
            else:
                self._write(f,'Man_n',self.man_n)

            if self.low_spot_distance is not None:
                if self.low_spot_is_meters:
                    self._write(f,'Low_Spot_Dist_m',self.low_spot_distance)
                else:
                    self._write(f,'Low_Spot_Range',self.low_spot_distance)
                if self.low_spot_use_box:
                    self._write(f,'Low_Spot_Range_Box')
                    self._write(f,'Low_Spot_Range_Box_Size',self.box_size)
            
            if self.find_flat:
                self._write(f,'Low_Spot_Find_Flat')
                if self.low_spot_find_flat_cutoff < float('inf'):
                    self._write(f,'Low_Spot_Range_FlowCutoff',self.low_spot_find_flat_cutoff)

            if self.ar_bathy:
                bathy_file = os.path.join(self.DATA_DIR, 'ar_bathy')
                os.makedirs(bathy_file, exist_ok=True)
                bathy_file = os.path.join(bathy_file, f"{os.path.basename(dem).split('.')[0]}__ar_bathy.txt")
                self._write(f,'BATHY_Out_File',bathy_file)
                self._write(f,'Bathymetry_Alpha',self.bathy_alpha)

                if self.bathy_method == 'Parabolic':
                    self._write(f,'Bathymetry_Method',0)
                elif self.bathy_method == 'Left Bank Quadratic':
                    self._write(f,'Bathymetry_Method', 1)
                    self._write(f,'Bathymetry_XMaxDepth',self.bathy_x_max_depth)
                    self._write(f,'Bathymetry_YShallow',self.bathy_y_shallow)
                elif self.bathy_method == 'Right Bank Quadratic':
                    self._write(f,'Bathymetry_Method', 2)
                    self._write(f,'Bathymetry_XMaxDepth',self.bathy_x_max_depth)
                    self._write(f,'Bathymetry_YShallow',self.bathy_y_shallow)
                elif self.bathy_method == 'Double Quadratic':
                    self._write(f,'Bathymetry_Method', 3)
                    self._write(f,'Bathymetry_XMaxDepth',self.bathy_x_max_depth)
                    self._write(f,'Bathymetry_YShallow',self.bathy_y_shallow)
                elif self.bathy_method == 'Trapezoidal':
                    self._write(f,'Bathymetry_Method', 4)
                    self._write(f,'Bathymetry_XMaxDepth',self.bathy_x_max_depth)
                else: self._write(f,'Bathymetry_Method', 5)

                if self.da_flow_param: self._write(f, 'RAPID_DA_or_Flow_Param',self.da_flow_param)

            f.write('\n')
            self._write(f,'# FloodSpreader Inputs')
            f.write('\n')

            if flowfile:
                id_flow_file = self._format_files(flowfile)
                self._warn_DNE('ID Flow File', id_flow_file)
                self._check_type('ID Flow File', id_flow_file, ['.txt','.csv'])
                self._write(f,'Comid_Flow_File',id_flow_file)

            if self.omit_outliers:
                if self.omit_outliers == 'Flood Bad Cells':
                    self._write(f,'Flood_BadCells')
                elif self.omit_outliers == 'Use AutoRoute Depths':
                    self._write(f,'FloodSpreader_Use_AR_Depths')
                elif self.omit_outliers == 'Smooth Water Surface Elevation':
                    self._write(f,'FloodSpreader_SmoothWSE_SearchDist',self.wse_search_dist)
                    self._write(f,'FloodSpreader_SmoothWSE_FractStDev',self.wse_threshold)
                    if self.wse_remove_three:
                        self._write(f,'FloodSpreader_SmoothWSE_RemoveHighThree')
                elif self.omit_outliers == 'Use AutoRoute Depths (StDev)':
                    self._write(f,'FloodSpreader_Use_AR_Depths_StDev')
                elif self.omit_outliers == 'Specify Depth' and self.specify_depth:
                    self._write(f,'FloodSpreader_SpecifyDepth',self.specify_depth)
                else:
                    logging.warning(f'Unknown outlier omission option: {self.omit_outliers}')

            self._write(f,'TopWidthDistanceFactor',self.twd_factor)
            if self.only_streams: self._write(f,'FloodSpreader_JustStrmDepths')
            if self.use_ar_top_widths: self._write(f,'FloodSpreader_Use_AR_TopWidth')
            if self.flood_local: self._write(f,'FloodLocalOnly')

            if self.DEPTH_MAP:
                depth_map = self._format_files(os.path.join(self.DEPTH_MAP, f"{os.path.basename(dem).split('.')[0]}__depth.tif"))
                if self.OVERWRITE or not os.path.exists(depth_map):
                    self._check_type('Depth Map',depth_map,['.tif'])
                    self._write(f,'OutDEP',depth_map)

            if self.FLOOD_MAP:
                flood_map = self._format_files(os.path.join(self.FLOOD_MAP, f"{os.path.basename(dem).split('.')[0]}__flood.tif"))
                if self.OVERWRITE or not os.path.exists(flood_map):
                    self._check_type('Flood Map',flood_map,['.tif'])
                    self._write(f,'OutFLD',flood_map)

            if self.VELOCITY_MAP:
                velocity_map = self._format_files(os.path.join(self.VELOCITY_MAP, f"{os.path.basename(dem).split('.')[0]}__vel.tif"))
                if self.OVERWRITE or not os.path.exists(velocity_map):
                    self._check_type('Velocity Map',velocity_map,['.tif'])
                    self._write(f,'OutVEL',velocity_map)

            if self.WSE_MAP:
                wse_map = self._format_files(os.path.join(self.WSE_MAP, f"{os.path.basename(dem).split('.')[0]}__wse.tif"))
                if self.OVERWRITE or not os.path.exists(wse_map):
                    self._check_type('WSE Map',wse_map,['.tif'])
                    self._write(f,'OutWSE',wse_map)

            if self.fs_bathy and self.ar_bathy: 
                fs_bathy_file = os.path.join(self.DATA_DIR, "bathy_fs")
                os.makedirs(fs_bathy_file, exist_ok=True)
                fs_bathy_file = os.path.join(fs_bathy_file, f"{os.path.basename(dem).split('.')[0]}__fs_bathy.tif")
                self._check_type('FloodSpreader Generated Bathymetry',fs_bathy_file,['.tif'])
                self._write(f,'FSOutBATHY', fs_bathy_file)
                if self.fs_bathy_smooth_method == 'Linear Interpolation':
                    self._write(f,'Bathy_LinearInterpolation')
                elif self.fs_bathy_smooth_method == 'Inverse-Distance Weighted':
                    self._write(f,'BathyTopWidthDistanceFactor', self.bathy_twd_factor)

        return mifn

    def _format_files(self,file_path: str) -> str:
        """
        Function added so that windows paths that pad with quotation marks can be valid
        """
        if any(file_path.endswith(s) and file_path.startswith(s) for s in {'"',"'"}):
            return file_path[1:-1]
        return file_path
        
    def _write(self,f, Card, Argument = '') -> None:
        if Argument:
            f.write(f"{Card}\t{Argument}\n")
        else:
            f.write(f"{Card}\n")

    def _warn_DNE(self, card: str, value: str) -> None:
        if value:
            if not os.path.exists(value):
                logging.warning(f"The file {value} for {card} does not exist: ")

    def _check_type(self, card: str, value: str, types: List[str]) -> None:
        if not any(value.endswith(t) for t in types):
            logging.warning(f"{card} is {os.path.basename(value)}, which does not have a valid file type ({','.join(types)})")
    
    def _zip_files(self, *args) -> Set[Tuple[str]]:
        """"
        Given a bunch of lists, sort each list so that the filenames match, and create a list of tuples 
        """
        # Get max length of all args
        max_len = max(len(a) for a in args)
        args = [sorted(a, key=os.path.basename) for a in args]
        output = set({})
        for i in range(max_len):
            out_tuple = []
            for arg in args:
                if i < len(arg):
                    out_tuple.append(arg[i])
                else:
                    out_tuple.append("")
            output.add(tuple(out_tuple))

        return output
    
    def run_autoroute(self, mifn: str) -> None:
        try:
            exe = self._format_files(self.AUTOROUTE.strip())
            mifn = self._format_files(mifn.strip())
            process = subprocess.run(f'conda activate {self.AUTOROUTE_CONDA_ENV} && echo "a" | {exe} {mifn}', # We echo a dummy input in so that AutoRoute can terminate if some input is wrong
                                     stdout=asyncio.subprocess.PIPE,
                                     stderr=asyncio.subprocess.PIPE,
                                     shell=True,)
        except Exception as e:
            logging.error(f"Error running autoroute: {e}")

        if process.returncode == 0:
            logging.info('Program finished')
            return process.stdout.decode('utf-8')
        logging.error(f"Error running autoroute: {process.stderr.decode('utf-8')}")
        return process.stderr.decode('utf-8')
