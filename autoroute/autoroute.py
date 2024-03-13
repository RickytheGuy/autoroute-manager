import logging
import sys
import os
import re
import glob
import multiprocessing
from typing import Tuple, Any

import geopandas as gpd
import pandas as pd
import numpy as np
import psutil
import yaml
from osgeo import gdal, osr

# GDAL setups
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = 'TRUE' # Set to avoid reading really large folders
os.environ["GDAL_NUM_THREADS"] = 'ALL_CPUS'

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class AutoRouteHandler:
    def __init__(self, 
                 ) -> None:
        self.setup()

    def run(self, yaml_file: str) -> None:
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

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            for key, value in data.items():
                setattr(self, key, value)
        self.DATA_DIR = ...
        slurm_scripts = []
        if self.BUFFER_FILES:
            if not all({self.DEM_FOLDER, self.DEM_NAME, self.STREAM_NETWORK_FOLDER, self.LAND_USE_FOLDER, self.FLOWFILE, self.ID_COLUMN, self.FLOW_COLUMN}):
                logging.error('You\'re missing some inputs for "Buffer Files"')
                exit()
            num_dems = self.find_number_of_dems_in_extent()
            slurm_scripts.append("""#!/bin/bash --login

#SBATCH --time=0:15:00   # walltime, 4 min per dem
#SBATCH --ntasks=10  # number of processor cores (i.e. tasks) - 30
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=3333M #memory per CPU core
#SBATCH -J "buffering_files"   # job name

mamba activate autoroute
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
""")
            
    def setup(self) -> None:
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
        if not DATA_DIR:
            logging.error('No working folder provided!')
        os.makedirs(DATA_DIR, exist_ok = True)
        os.chdir(DATA_DIR)
        with open('This is the working folder. Please delete and modify with caution.txt', 'a') as f:
            pass

        os.makedirs('dems_buffered', exist_ok = True)
        os.makedirs('stream_files', exist_ok = True)
        os.makedirs('lu_bufferd', exist_ok = True)
        os.makedirs('rapid_files', exist_ok = True)
        os.makedirs('flow_files', exist_ok = True)
        os.makedirs('vdts', exist_ok = True)
        os.makedirs('mifns', exist_ok = True)
        os.makedirs('stream_reference_files', exist_ok=True)

    def buffer(self) -> None:
        pass

    def find_dems_in_extent(self, 
                            extent = EXTENT) -> list[str]:
        dems = self.find_files(DEM_FOLDER)
        if extent:
            if dem_file_pattern:
                dems = [dem for dem in dems if self.is_in_extent_re(dem, extent)]
            else:
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
    
    def is_in_extent_re(self, 
                        dem: str,
                        extent = EXTENT) -> bool:
        minx1, miny1, maxx1, maxy1 = extent 
        try:
            if DEM_NAME == 'FABDEM': # We use FABDEM alot, thus it is built in
                file_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(dem_file_pattern, os.path.basename(dem))[0]]
                file_extent.reverse()
                file_extent += [file_extent[0] + 1, file_extent[1] + 1]
                minx2, miny2, maxx2, maxy2 = file_extent
                if (minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2):
                    return True
                return False
            else:
                raise NotImplementedError()
        except Exception as e:
            logging.error(f"Bad dem file pattern! Error:\n{e}")
            return False

    def is_in_extent_gdal(self, 
                          dem: str,
                          extent=EXTENT) -> bool:
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

        os.makedirs(os.path.join(DATA_DIR, 'dems_buffered', DEM_NAME), exist_ok=True)
        buffered_dem = os.path.join(DATA_DIR, 'dems_buffered', DEM_NAME, f"{str(round(minx, 3)).replace('.','_')}__{str(round(miny, 3)).replace('.','_')}__{str(round(maxx, 3)).replace('.','_')}__{str(round(maxy, 3)).replace('.','_')}.vrt")
        if OVERWRITE and os.path.exists(buffered_dem):
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
    
    def create_strm_file(self, dem: str) -> None:
        ds = self.open_w_gdal(dem)
        if not ds: return
        projection = ds.GetProjection()

        strm = os.path.join(DATA_DIR, 'stream_files', f"{DEM_NAME}__{STREAM_NAME}")
        os.makedirs(strm, exist_ok=True)
        strm = os.path.join(strm, f"{os.path.basename(dem).split('.')[0]}__strm.tif")
        if OVERWRITE and os.path.exists(strm):
            logging.info(f"{strm} already exists. Skipping...")
            return

        strm_reference = os.path.join(DATA_DIR, 'stream_reference_files', f"{STREAM_NAME}_ref.parquet")
        if not os.path.exists(strm_reference):
            self.create_reference_file(strm_reference)

        minx, miny, maxx, maxy = self.get_ds_extent(ds)
        ref_df = pd.read_parquet(strm_reference)
        ref_df = ref_df[
            (ref_df['minx'] >= minx) & 
            (ref_df['miny'] >= miny) &
            (ref_df['maxx'] <= maxx) &
            (ref_df['maxy'] <= maxy)
        ]
        if ref_df.empty:
            logging.warning('No streams found in the DEM\'s extent!')
            return
        
        filenames = [os.path.join(STREAM_NETWORK_FOLDER, f) for f in np.unique(ref_df['filename'].values)]
        ref_df = None
        if len(filenames) > 1:
            dfs = []
            for f in filenames:
                if os.path.basename(f).endswith(('.parquet', '.geoparquet')):
                    dfs.append(gpd.read_parquet(f, columns=[ID_COLUMN, 'geometry']))
                else:
                    dfs.append(gpd.read_file(f))
            pd.concat(dfs, ignore_index=True).to_file('temp.gpkg')
            file = 'temp.gpkg'
        else:
            file = filenames[0]
            if file.endswith('.parquet'):
                gpd.read_parquet(file, columns=[ID_COLUMN, 'geometry']).to_file('temp.gpkg')
                file = 'temp.gpkg'
        


        options = gdal.RasterizeOptions(attribute=ID_COLUMN,
                              outputType=gdal.GDT_UInt64,
                              format='GTiff',
                              outputSRS=projection,
                              creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
                              outputBounds=(minx, miny, maxx, maxy),
                              noData=0,
                              width=ds.RasterXSize,
                              height=ds.RasterYSize)
        gdal.Rasterize(strm, file, options=options)
        2


        
        
    def create_reference_file(self, output_ref_file: str):
        # Each cpu needs 4x max memory of biggest file
        stream_files = [f for f in self.find_files(STREAM_NETWORK_FOLDER, '*.*') if f.endswith(('.gpkg','.shp','.parquet'))]
        if not stream_files:
            logging.warning(f'No stream files found in {STREAM_NETWORK_FOLDER}')
            exit()
        biggest_file = os.path.getsize(max(stream_files, key = os.path.getsize))
        available_memory = psutil.virtual_memory().total * 0.85 # Allow machine to consume 85% memory
        processes = min(len(stream_files), multiprocessing.cpu_count(), available_memory // (biggest_file * 2))
        worker_lists = self.list_to_sublists(stream_files, processes)
        
        with multiprocessing.Pool(processes=processes) as pool:
            dfs = pool.map(self.reference_file_helper, worker_lists)
        dfs = [d for sub in dfs for d in sub]
        
        pd.concat(dfs, ignore_index=True).to_parquet(output_ref_file)
            
    def reference_file_helper(self,stream_files: list[str]) -> pd.DataFrame:
        dfs = []
        for strm in stream_files:
            if os.path.basename(strm).endswith(('.parquet', '.geoparquet')):
                df = gpd.read_parquet(strm, columns=[ID_COLUMN, 'geometry'])
            else:
                df = gpd.read_file(strm)
            df['filename'] = os.path.basename(strm)
            dfs.append(pd.concat([df[[ID_COLUMN, 'filename']], df.bounds], axis=1))
        return dfs

    def list_to_sublists(self, alist: list[Any], n: int) -> list[list[Any]]:
        return [alist[x:x+n] for x in range(0, len(alist), n)]
        

if __name__ == "__main__":
    gdal.UseExceptions()
    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"
    # args = sys.argv
    # if len(args) == 1:
    #     logging.info('No inputs given. Exiting...')
    #     exit()
    # mh = MasterHandler(mode=args[1])
    mh = AutoRouteHandler()
    #mh.buffer_dem('/home/lrr43/fsl_groups/grp_geoglows2/compute/fabdem/DEMs_for_Entire_World/N00E020-N10E030_FABDEM_V1-2/N00E022_FABDEM_V1-2.tif')
    mh.create_strm_file('/home/lrr43/fsl_groups/grp_geoglows2/compute/fabdem/DEMs_for_Entire_World/N00E000-N10E010_FABDEM_V1-2/N00E006_FABDEM_V1-2.tif')
    logging.info('Finished')
    
