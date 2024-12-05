
import os
import json
import platform
import re
import threading
from typing import Tuple, Dict, List

import multiprocessing
import pyogrio
import contextily as ctx
import pyproj.exceptions
import xarray as xr
import gradio as gr
import numpy as np
import pyproj
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import geopandas as gpd

from osgeo import gdal, ogr
from shapely.geometry import box
from pyproj import Transformer

from autoroute_manager.autoroute import AutoRoute
from autoroute_manager import LOG


gdal.UseExceptions()

SYSTEM = platform.system()

class ManagerFacade():
    map_df = None
    map_extents = None
    strm_hash = None
    def init(self) -> None:
        self.manager = AutoRoute(None)
        self.docs_file = 'defaults_and_docs.json'
        self._strm_info_file = os.path.join(os.path.dirname(__file__), '.files_info.json')
        self._strm_info_dict = {}
        if os.path.exists(self._strm_info_file):
            with open(self._strm_info_file, 'r') as f:
                try:
                    tmp = json.load(f)
                except json.JSONDecodeError:
                    tmp = {}
                self._strm_info_dict = dict(tmp)

        docs = None
        data = None
        if not os.path.exists(self.docs_file):
            self.docs_file = os.path.join(os.path.dirname(__file__), 'defaults_and_docs.json')
        if not os.path.exists(self.docs_file):
            gr.Warning('Could not find the defaults_and_docs.json file')
        else:
            with open(self.docs_file, 'r', encoding='utf-8') as f:
                d = json.loads(f.read())[0]
                docs: Dict[str,str] = d['docs']
                data: Dict[str,str] = d['data']
        self.docs = docs
        self.data = data
        
    async def run(self, **kwargs):
        await self._run(**kwargs)

    def doc(self, key: str) -> str:
        return self.docs.get(key, 'No documentation available')
    
    def default(self, key: str) -> str:
        return self.data.get(key, None)
    
    def get_ids(self, *args) -> None:
        LOG.info("Getting IDs...")
        dem = args[0]
        strm_lines = self._format_files(args[1])
        minx = args[2]
        miny = args[3]
        maxx = args[4]
        maxy = args[5]
        ids_folder = self._format_files(args[6])
        flow_id = args[7]
        
        if not ids_folder:
            msg = "No output folder specified"
            LOG.error(msg)
            gr.Error(msg)
            return
        if os.path.isdir(strm_lines):
            stream_files = [os.path.join(strm_lines,f) for f in os.listdir(strm_lines) if f.endswith(('.shp', '.gpkg', '.parquet', '.geoparquet'))]
        elif os.path.isfile(strm_lines):
            stream_files = [strm_lines]
        else:
            msg = f"{strm_lines} is not a valid file or folder"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if os.path.isdir(ids_folder):
            output_file = os.path.join(ids_folder, 'ids.csv')
        else:
            output_file = ids_folder
        num_processes = min(os.cpu_count(), len(stream_files))
        with multiprocessing.Pool(num_processes) as pool:
            try:
                results = pool.starmap(self._get_ids, [(f, [minx, miny, maxx, maxy], flow_id) for f in stream_files])
            except KeyError:
                msg = f"{flow_id !r} is not a valid field in the stream files provided"
                LOG.error(msg)
                gr.Error(msg)
                return
        if len(results) == 0:
            msg = "No IDs found in the stream files provided"
            LOG.error(msg)
            gr.Error(msg)
            return
        if len(results) == 1:
            results = np.unique(results[0])
        else:
            results = np.unique(np.concatenate(results, axis=None))
        pd.DataFrame({flow_id: results}, dtype=int).to_csv(output_file, index=False)
        msg = f"IDs saved to {output_file}"
        LOG.info(msg)
        gr.Info(msg)
        
        
    def _get_ids(self, strm_file: str, extent: List[float], id_field: str) -> np.ndarray:
        data = pyogrio.read_info(strm_file)

        if not data['crs']:
            if np.abs(np.asarray(data['total_bounds'])).max() <= 180:
                crs_epsg = 4326
            else:
                msg = f"Could not determine the CRS for {strm_file}"
                LOG.error(msg)
                gr.Error(msg)
                return np.array([])
        else:
            crs_epsg = int(data['crs'].split(':')[1])

        f_extent = box(*data['total_bounds'])
        if crs_epsg != 4326:
            transformer = Transformer.from_crs("EPSG:4326",f"EPSG:{crs_epsg}", always_xy=True) 
            minx2, miny2 = transformer.transform(extent[0], extent[1])
            maxx2, maxy2 =  transformer.transform(extent[2], extent[3])
            bbox = box(minx2, miny2, maxx2, maxy2)
        else:
            bbox = box(*extent)
        if not f_extent.intersects(bbox):
            return np.array([])

        gdf = self.manager.gpd_read(strm_file, columns=[id_field], bbox=bbox)
        return gdf[id_field].to_numpy().astype(int)
        
    
    def save(self, *args) -> str:
        """
        Save the modified documentation and defaults

        order of args: 
        0. dem, curve_file, strm_lines, id_flow_id_col, lu_file, 
        5. base_max_id_col, base_max_file, subtract_baseflow, streamlines_id_col, flow_params, 
        10. flow_baseflow, num_iterations,meta_file, convert_cfs_to_cms, x_distance, 
        15. q_limit, LU_Manning_n, direction_distance, slope_distance, low_spot_distance, 
        20. low_spot_is_meters,low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, 
        25. degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, 
        30. row_end, use_prev_d_4_xs,weight_angles, man_n, adjust_flow, 
        35. bathy_alpha, ar_bathy_out_file, id_flow_file, omit_outliers, wse_search_dist, 
        40. wse_threshold, wse_remove_three,specify_depth, twd_factor, only_streams, 
        45. use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, 
        50. wse_map, fs_bathy_file, da_flow_param,bathy_method,bathy_x_max_depth, 
        55. bathy_y_shallow, fs_bathy_smooth_method, bathy_twd_factor,data_dir, minx, 
        60. miny, maxx, maxy, overwrite, buffer, 
        65. crop, vdt_file,  ar_exe, fs_exe, clean_outputs
        70. buffer_distance, use_ar_python, run_ar_bathy
        """
        if len(args) < 69: 
            LOG.warning('Could not save parameters')
            return ""
        to_write = [
            {
                "docs": self.docs,
                "data": {
                    "DEM":args[0],
                    "curve_file":args[1],
                    "strm_lines":args[2],
                    "id_flow_id_col":args[3],
                    "base_max_id_col":args[5],

                    "streamlines_id_col":args[8],
                    "ar_exe":args[67],
                    "fs_exe":args[68],
                    "x_distance":args[14],
                    "q_limit":args[15],
                    "use_prev_d_4_xs":args[31],
                    "man_n":args[33],
                    "lu_file":args[4],

                    "LU_Manning_n":args[16],
                    "base_max_file":args[6],
                    "Gen_Dir_Dist":args[17],
                    "Gen_Slope_Dist":args[18],
                    "Weight_Angles":args[32],

                    "degree_manip":args[25],
                    "degree_interval":args[26],
                    "Low_Spot_Range":args[19],
                    "Low_Spot_Find_Flat_Cutoff":args[24],
                    "vdt":args[66],
                    "num_iterations":args[11],
                    "flow_params_ar":args[9],
                    "flow_baseflow":args[10],
                    "subtract_baseflow":args[7],
                    "Bathymetry_Alpha":args[35],

                    "ADJUST_FLOW_BY_FRACTION":args[34],
                    "BATHY_Out_File":args[36],
                    "Meta_File":args[12],
                    "Comid_Flow_File":args[37],
                    "fs_bathy_file":args[51],
                    "omit_outliers":args[38],
                    "FloodSpreader_SpecifyDepth":args[42],
                    "twd_factor":args[43],
                    "only_streams":args[44],
                    "use_ar_top_widths":args[45],
                    "FloodLocalOnly":args[46],
                    "out_depth":args[47],
                    "out_flood":args[48],
                    "out_velocity":args[49],
                    "out_wse":args[50],
                    "RAPID_DA_or_Flow_Param":args[52],
                    "bathy_method":args[53],
                    "bathy_x_max_depth":args[54],
                    "bathy_y_shallow":args[55],
                    "overwrite":args[63],
                    "buffer":args[64],
                    "crop":args[65],
                    "data_dir":args[58],
                    "minx":args[59],
                    "maxx":args[61],
                    "miny":args[60],
                    "maxy":args[62],
                    "convert_cfs_to_cms":args[13],
                    "low_spot_is_meters":args[20],
                    "low_spot_use_box":args[21],
                    "box_size":args[22],
                    "find_flat":args[23],
                    "bathy_twd_factor":args[57],
                    "fs_bathy_smooth_method":args[56],
                    "clean_outputs":args[69],
                    "buffer_distance":args[70],
                    "use_ar_python":args[71],
                    "run_ar_bathy":args[72]
                }
            }
        ]

        with open(self.docs_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(to_write, indent=4))
            
        LOG.info("Parameters saved!")
        gr.Info("Parameters saved!")
        return ""

    #@staticmethod
    def _format_files(self,file_path: str) -> str:
        """
        Function added so that windows paths that pad with quotation marks can be valid
        """
        if not file_path: return ""
        if any(file_path.startswith(s) for s in ['"',"'"]):
            file_path = file_path[1:]
        if any(file_path.endswith(s) for s in ['"',"'"]):
            file_path = file_path[:-1]
        return file_path

    def _write(self,f, Card, Argument = '') -> None:
        if Argument:
            f.write(f"{Card}\t{Argument}\n")
        else:
            f.write(f"{Card}\n")

    def _warn_DNE(self, card: str, value: str) -> None:
        if value:
            if not os.path.exists(value):
                gr.Warning(f"The file {value} for {card} does not exist: ")

    def _check_type(self, card: str, value: str, types: List[str]) -> None:
        if not any(value.endswith(t) for t in types):
            gr.Warning(f"{card} is {os.path.basename(value)}, which does not have a valid file type ({','.join(types)})")

    def omit_outliers_change(self,value: str) -> Tuple[gr.Radio, gr.Column, gr.Number]:
        if value == 'None':
            return gr.Radio(info='None: No outliers will be removed'), gr.Column(visible=False), gr.Number(visible=False)
        elif value == 'Flood Bad Cells':
            return gr.Radio(info=self.docs['Flood_BadCells']), gr.Column(visible=False), gr.Number(visible=False)
        elif value == 'Use AutoRoute Depths':
            return gr.Radio(info=self.docs['FloodSpreader_Use_AR_Depths']), gr.Column(visible=False), gr.Number(visible=False)
        elif value == 'Smooth Water Surface Elevation':
            return  gr.Radio(info=self.docs['smooth_wse']), gr.Column(visible=True), gr.Number(visible=False)
        elif value == 'Use AutoRoute Depths (StDev)':
            return gr.Radio(info=self.docs['FloodSpreader_Use_AR_Depths_StDev']), gr.Column(visible=False), gr.Number(visible=False)
        else:
            return gr.Radio(info=self.docs['FloodSpreader_SpecifyDepth']), gr.Column(visible=False), gr.Number(visible=True)

    def bathy_changes(self,value: str, use_python:bool) -> Tuple[gr.Slider, gr.Slider]:
        if value == 'Trapezoidal' or use_python:
            return gr.Slider(visible=True), gr.Slider(visible=False)
        if value in ['Left Bank Quadratic', 'Right Bank Quadratic', 'Double Quadratic']:
            return gr.Slider(visible=True), gr.Slider(visible=True)
        return gr.Slider(visible=False), gr.Slider(visible=False)

    def show_mans_n(self,lu, mannings) -> gr.Number:
        """
        Let mannings n input box be interactive if these two are both not specified
        """
        if lu and mannings:
            return gr.Number(interactive=False)
        return gr.Number(interactive=True)

    def update_flow_params(self,flow_file) -> Tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
        flow_file = self._format_files(flow_file)
        if not os.path.exists(flow_file):
            return None, None, None, None
        try:
            if flow_file.lower().endswith(('.csv', '.txt')):
                cols = sorted(list(pd.read_csv(flow_file, delimiter=',').columns))
                if len(cols) <= 1:
                    try:
                        cols = sorted(list(pd.read_csv(flow_file, delimiter=' ').columns))
                    except:
                        return None, None, None, None
            elif flow_file.lower().endswith(('.nc', '.nc3','.nc4')):
                ds = xr.open_dataset(flow_file)
                cols = sorted(list(ds.data_vars))
        except:
                return None, None, None, None

        return gr.Dropdown(choices=cols), gr.Dropdown(choices=cols), gr.Dropdown(choices=cols), gr.Dropdown(choices=cols)

    def dem_mods_change(self,value: str) -> Tuple[gr.Markdown, gr.DataFrame, gr.Textbox]:
        'Crop to Extent', 'Clip with Mask'
        if value == 'Crop to Extent':
            return gr.Markdown(visible=True), gr.DataFrame(visible=True), gr.Textbox(visible=False)
        elif value == 'Clip with Mask':
            return gr.Markdown(visible=False), gr.DataFrame(visible=False), gr.Textbox(visible=True)
        else:
            return gr.Markdown(visible=False), gr.DataFrame(visible=False), gr.Textbox(visible=False)

    async def _run(self,dem, curve_file, strm_lines, id_flow_id_col, lu_file, base_max_id_col, base_max_file, subtract_baseflow, streamlines_id_col, flow_params_ar, flow_baseflow, num_iterations,
                                                    meta_file, convert_cfs_to_cms, x_distance, q_limit, mannings_table, direction_distance, slope_distance, low_spot_distance, low_spot_is_meters,
                                                    low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, row_end, use_prev_d_4_xs,
                                                    weight_angles, man_n, adjust_flow, bathy_alpha, ar_bathy, id_flow_file, omit_outliers, wse_search_dist, wse_threshold, wse_remove_three,
                                                    specify_depth, twd_factor, only_streams, use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, wse_map, fs_bathy_file, da_flow_param,
                                                    bathy_method,bathy_x_max_depth, bathy_y_shallow, fs_bathy_smooth_method, bathy_twd_factor,
                                                    data_dir, minx, miny, maxx, maxy, overwrite, buffer, crop, vdt_file,  ar_exe, fs_exe, clean_outputs, buffer_distance, use_ar_python, run_ar_bathy) -> str:
        """
        Write the main input file
        """
        if use_ar_python:
            if not lu_file:
                msg = "Please specify a land use file when using ARC"
                gr.Error(msg)
                LOG.error(msg)
                return ""
            if not mannings_table:
                msg = "Please specify a Mannings Table when using ARC"
                gr.Error(msg)
                LOG.error(msg)
                return ""
        gr.Info("Beginning model run!")
        if {minx, miny, maxx, maxy} == {0}:
            extent = None
        else:
            extent = (minx, miny, maxx, maxy)
        params = {"OVERWRITE": overwrite,
                  "DATA_DIR": self._format_files(data_dir),
                  "DEM_FOLDER": self._format_files(dem),
                  "BUFFER_FILES": buffer,
                  "BUFFER_DISTANCE": buffer_distance,
                  "STREAM_NETWORK_FOLDER": self._format_files(strm_lines),
                  "STREAM_ID": streamlines_id_col,
                  "SIMULATION_FLOWFILE": self._format_files(base_max_file),
                  "FLOOD_FLOWFILE":self._format_files(id_flow_file),
                  "SIMULATION_ID_COLUMN": id_flow_id_col,
                  "SIMULATION_FLOW_COLUMN": flow_params_ar,
                  "BASE_FLOW_COLUMN": flow_baseflow,
                  "BASE_MAX_ID_COLUMN": base_max_id_col,
                  "EXTENT": extent,
                  "CROP": crop,
                  "LAND_USE_FOLDER": self._format_files(lu_file),
                  "MANNINGS_TABLE": self._format_files(mannings_table),
                  "DEPTH_MAP": self._format_files(depth_map),
                  "FLOOD_MAP": self._format_files(flood_map),
                  "VELOCITY_MAP": self._format_files(velocity_map),
                  "WSE_MAP": self._format_files(wse_map),
                  "CLEAN_OUTPUTS": clean_outputs,
                  
                  "USE_PYTHON": use_ar_python,
                  "AUTOROUTE": self._format_files(ar_exe),
                  "FLOODSPREADER": self._format_files(fs_exe),
                  "AUTOROUTE_CONDA_ENV": "autoroute",

                    "curve_file": self._format_files(curve_file),
                  "RAPID_Subtract_BaseFlow": subtract_baseflow,
                  "VDT": self._format_files(vdt_file),
                  "num_iterations" : num_iterations,
                  "convert_cfs_to_cms": convert_cfs_to_cms,  
                  "x_distance": x_distance,
                  "q_limit": q_limit,
                  "direction_distance": direction_distance,
                  "slope_distance": slope_distance,
                  "weight_angles": weight_angles,
                  "use_prev_d_4_xs": use_prev_d_4_xs,
                  "adjust_flow": adjust_flow,

                  "degree_manip": degree_manip,
                  "degree_interval": degree_interval,
                  "man_n": man_n,
                  "low_spot_distance": low_spot_distance,
                  "low_spot_is_meters": low_spot_is_meters,
                  "low_spot_use_box": low_spot_use_box,
                  "box_size": box_size,
                  "find_flat": find_flat,
                  "low_spot_find_flat_cutoff": low_spot_find_flat_cutoff,
                  "run_bathymetry": run_ar_bathy,
                  "ar_bathy_file": self._format_files(ar_bathy),
                  "bathy_alpha": bathy_alpha,
                  "bathy_method": bathy_method,
                  "bathy_x_max_depth": bathy_x_max_depth,
                  "bathy_y_shallow":  bathy_y_shallow,
                  "da_flow_param": da_flow_param,
                  "omit_outliers": omit_outliers,
                  "wse_search_dist": wse_search_dist,
                  "wse_threshold": wse_threshold,
                  "wse_remove_three": wse_remove_three,
                  "specify_depth": specify_depth,
                  "twd_factor": twd_factor,
                  "only_streams": only_streams,
                  "use_ar_top_widths": use_ar_top_widths,
                  "flood_local": flood_local,
                  "fs_bathy_file": self._format_files(fs_bathy_file),
                  "fs_bathy_smooth_method": fs_bathy_smooth_method,
                  "bathy_twd_factor": bathy_twd_factor,
        }
        self.manager.setup(params)
        self.manager.run()
        gr.Info("Finished!")
        return ""
        
    def _prepare_exe(self,exe: str) -> str:
        if SYSTEM == "Windows":
            return exe
        if SYSTEM == "Darwin" or SYSTEM == "Linux": # MAC
            slashes = exe.count('/')
            if slashes == 0:
                return './' + exe
            if slashes == 1 and exe[0] == '/':
                return '.' + exe
        return exe

    def _sizeof_fmt(self, num:int) -> str:
        """
        Take in an int number of bytes, outputs a string that is human readable
        """
        for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
            if abs(num) < 1024.0:
                return f"{num:3.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} YB"
    
    def read_for_map(self, minx, miny, maxx, maxy, strm_lines) -> gpd.GeoDataFrame:
        if minx is None:
            minx = 39.5
        if miny is None:
            miny = 39
        if maxx is None:
            maxx = -111
        if maxy is None:
            maxy = -110

        if {minx, miny, maxx, maxy} == {0}: # If all are 0, then we have no data
            gr.Info("0 means no extent was specified")
            return None

        if maxx <= minx:
            gr.Warning('Max X is less than Min X')
            return None
        if maxy <= miny:
            gr.Warning('Max Y is less than Min Y')
            return None
        if abs(miny) > 90  or abs(maxy) > 90 or maxx > 180 or minx < -180:
            gr.Warning('Invalid coordinates')
            return None

        if not strm_lines:
            return None
        if os.path.isdir(strm_lines):
            stream_files = [os.path.join(strm_lines,f) for f in os.listdir(strm_lines) if f.endswith(('.shp', '.gpkg', '.parquet', '.geoparquet'))]
        elif os.path.isfile(strm_lines):
            stream_files = [strm_lines]
        else:
            msg = f"Could  not figure out what {stream_files} is"
            gr.Error(msg)
            LOG.error(msg)
            return None
        
        if not stream_files:
            return None
        
        if self.map_df is not None and self.map_extents == [minx, miny, maxx, maxy] and self.strm_hash == hash(strm_lines):
            return self.map_df
        
        a_file = stream_files[0]
        data: dict = pyogrio.read_info(a_file)
        minx1, miny1, maxx1, maxy1 = minx, maxx, miny, maxy
        if data.get('crs', False):
            try:
                data_crs = pyproj.CRS.from_wkt(data['crs'])
            except pyproj.exceptions.CRSError:
                data_crs = pyproj.CRS(data['crs'])
            if data_crs.is_exact_same(pyproj.CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")):
                pass
            else:
                # Transform minx, miny, maxx, maxy to the crs of the file
                transformer = Transformer.from_crs("EPSG:4326",data['crs'], always_xy=True)
                minx1, miny1 = transformer.transform(minx, miny)
                maxx1, maxy1 = transformer.transform(maxx, maxy)
        else:
            if np.abs(np.asarray(data['total_bounds'])).max() <= 180:
                pass # Assume it's in WGS84
            else:
                gr.Warning(f"Could not determine the CRS for files in {strm_lines}")
                return None
            
        dfs = []
        save = False
        for f in stream_files:
            if f in self._strm_info_dict:
                crs_epsg, extent = self._strm_info_dict[f]
                if not self.in_extent(crs_epsg, *extent, [minx1, maxx1, miny1, maxy1]):
                    continue
            else:
                ds: ogr.DataSource = ogr.Open(f)
                layer: ogr.Layer = ds.GetLayer()
                spatial_ref = layer.GetSpatialRef()
                if spatial_ref:
                    crs_epsg = spatial_ref.GetAuthorityCode("PROJCS") if spatial_ref.IsProjected() else spatial_ref.GetAuthorityCode("GEOGCS")
                else:
                    crs_epsg = 4326   
                crs_epsg = int(crs_epsg)
                _minx, _maxx, _miny, _maxy = layer.GetExtent()
                self._strm_info_dict[f] = [crs_epsg, [_minx, _miny, _maxx, _maxy]]
                save = True

                if not self.in_extent(crs_epsg, minx1, miny1, maxx1, maxy1, [_minx, _miny, _maxx, _maxy]):
                    continue

            if data['driver'].lower() == 'parquet':
                temp = gpd.read_parquet(f)
                temp = temp.cx[minx1:maxx1, miny1:maxy1]
                dfs.append(temp)
            else:
                dfs.append(gpd.read_file(f, bbox=(minx1, miny1, maxx1, maxy1)))

        if save: self.save_strm_info()

        df = gpd.GeoDataFrame(pd.concat(dfs))
        if df.crs is None:
            if df.bounds.max().max() <= 180:
                df = df.set_crs(epsg=4326)
            else:
                gr.Warning(f"Could not determine the CRS for files in {strm_lines}")
                return None
            
        df = df.to_crs(4326)
            
        self.map_df = df
        self.map_extents = [minx, miny, maxx, maxy]
        self.strm_hash = hash(strm_lines)
        return df

    def make_map(self, 
                 minx: float, 
                 miny: float, 
                 maxx: float, 
                 maxy: float,
                 strm_lines: str):
        strm_lines = self._format_files(strm_lines)
        if self.map_df is not None:
            dfs = [self.map_df, gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=4326)]
        else:
            thread = threading.Thread(target=self.read_for_map, args=(minx, miny, maxx, maxy, strm_lines))
            thread.start()
            dfs = [gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=4326)]
        dif = (((maxx - minx) + (maxy - miny)) / 2 ) * 0.25
        
        # Make some geometry to plot
        df: gpd.GeoDataFrame = pd.concat(dfs)

        plt.rcParams['figure.figsize'] = [12, 6]
        fig = plt.figure(edgecolor='black')
        ax = fig.add_axes([0, 0, 1, 1])
        df.plot(ax=ax, facecolor='none', edgecolor='blue')
        ax.set_xlim(max(minx - dif, -180), min(maxx + dif, 180))
        ax.set_ylim(max(miny - dif, -90), min(maxy + dif, 90))
        ax.set_xticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_yticks([])
        ax.tick_params(left = False, bottom = False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))

        ctx.add_basemap(ax, 
                    crs=4326, 
                    attribution=False, 
                    source="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                    )
        
        return fig
    
    def in_extent(self, crs_epsg: int, minx: float, miny: float, maxx: float, maxy: float, extent: list) -> bool:
        f_extent = box(*extent)
        if crs_epsg != 4326 and crs_epsg is not None:
            transformer = Transformer.from_crs("EPSG:4326",f"EPSG:{crs_epsg}", always_xy=True) 
            minx2, miny2 = transformer.transform(minx, miny)
            maxx2, maxy2 =  transformer.transform(maxx, maxy)
            bbox = box(minx2, miny2, maxx2, maxy2)
        else:
            bbox = box(minx, miny, maxx, maxy)
        return f_extent.intersects(bbox)

    def make_fancy_map(self, 
                 minx: float, 
                 miny: float, 
                 maxx: float, 
                 maxy: float,
                 strm_lines: str) -> go.Figure:
        strm_lines = self._format_files(strm_lines)
        df = self.read_for_map(minx, miny, maxx, maxy, strm_lines)
        if df is None or df.empty:
            return ""
                
        cols = set(df.columns)
        id_col = [c for c in cols if c.lower() in ['linkno', 'comid', 'streamid', 'stream_id', 'tdxhydrolinkno', 'rivid', 'river_id', 'id']]
        if id_col:
            id_col = id_col[0]
            df = df.set_index(id_col)
        else:
            id_col = 'id'

        df = df.get_coordinates().rename_axis(id_col).groupby(id_col).agg(['first','last'])

        lons = np.empty(3 * len(df))
        lons[::3] = df[('x', 'first')].values
        lons[1::3] = df[('x', 'last')].values
        lons[2::3] = None
        lons = np.append(lons, [minx, maxx, maxx, minx, minx, None])

        lats = np.empty(3 * len(df))
        lats[::3] = df[('y', 'first')].values
        lats[1::3] = df[('y', 'last')].values
        lats[2::3] = None
        lats = np.append(lats, [miny, miny, maxy, maxy, miny, None])

        names = np.empty(3 * len(df))
        names[::3] = df.index.values
        names[1::3] = df.index.values
        names[2::3] = None
        names = np.append(names, ['Extent', 'Extent', 'Extent', 'Extent', 'Extent', None])


        fig = px.line_map(lat=lats, lon=lons, hover_name=names, center={'lat': (miny + maxy) / 2, 'lon': (minx + maxx) / 2}, zoom=9 )
        fig.update_layout(
            hovermode='closest',
            map_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        #"https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                        "http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}"
                    ]
                }
            ],
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        fig.update_traces(line=dict(width=5))
        fig.show()
        return ""
        
    def get_forecast(self, input_path: str, output_path: str, date: str, ensemble: str) -> None:
        try:
            import geoglows
        except ImportError:
            msg = "Please install the geoglows package to use this function. Run 'conda install geoglows' in your terminal."
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not os.path.exists(input_path):
            msg = f"{input_path} does not exist"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not output_path:
            msg = "No output path specified"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not ensemble:
            msg = "No ensemble specified"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not date or not re.match(r'\d{4}\d{2}\d{2}', date):
            msg = "Invalid date format. Please use the format 'YYYYMMDD'"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        # Check if the file has a header or not
        with open(input_path, 'r') as f:
            first_line = f.readline()
        if first_line[0].isdigit():
            header = None
        else:
            header = 0
        
        # Read the CSV file
        df = pd.read_csv(input_path, header=header)
        
        # Save the first column as a list of IDs and remove duplicates
        ids = np.unique(df.iloc[:,0].values)
        
        forecast_data = geoglows.data.forecast_ensembles(ids, date=date)
        
        #reset the index
        forecast_data.reset_index(inplace=True)
        forecast_data = forecast_data[['time','river_id',ensemble]]
        max_rows: pd.DataFrame = forecast_data.loc[forecast_data.groupby('river_id')[ensemble].idxmax()]
    
        max_rows.drop(columns=['time'], inplace=True)
        
        # Reset the index
        max_rows.reset_index(drop=True, inplace=True)
        
        # Rename the 'ensemble_52' column to 'max_forecast'
        max_rows.rename(columns={ensemble: 'max_forecast'}, inplace=True)
        max_rows.rename(columns={'river_id': 'LINKNO'}, inplace=True)
        
        max_rows.to_csv(output_path, index=False)
        
    def get_median_max_forecast(self, input_path: str, output_path: str, date: str) -> None:
        try:
            import geoglows
        except ImportError:
            msg = "Please install the geoglows package to use this function. Run 'conda install geoglows' in your terminal."
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not os.path.exists(input_path):
            msg = f"{input_path} does not exist"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not output_path:
            msg = "No output path specified"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        if not date or not re.match(r'\d{4}\d{2}\d{2}', date):
            msg = "Invalid date format. Please use the format 'YYYYMMDD'"
            LOG.error(msg)
            gr.Error(msg)
            return
        
        # Read the CSV file
        df = open_csv_regardless_of_header(input_path)
        
        # Save the first column as a list of IDs and remove duplicates
        ids = np.unique(df.iloc[:,0].values)
        
        LOG.info(f"Getting forecast data for {len(ids)} rivers")
        gr.Info(f"Getting forecast data for {len(ids)} rivers")
        
        num_to_run = min(os.cpu_count(), len(ids))
        
        ids_date = self.list_to_sublists(ids, num_to_run, date)
        with multiprocessing.Pool(num_to_run) as pool:
            max_median_flows = pool.starmap(help_get_forecast_median, ids_date)
            
        max_median_flows = np.array(max_median_flows).flatten()
        
        pd.DataFrame({'LINKNO': ids, 'max_median_flow': max_median_flows}).to_csv(output_path, index=False)
        msg = "Finished getting forecast median flow data"
        gr.Info(msg)
        LOG.info(msg)
            
    def list_to_sublists(self, alist: List, n: int, add_tuple=None) -> List:
        if add_tuple:
            return [(alist[x:x+n], add_tuple) for x in range(0, len(alist), n)]
        
        return [alist[x:x+n] for x in range(0, len(alist), n)]
    
    def save_strm_info(self) -> None:
        with open(self._strm_info_file, 'w') as f:
            json.dump(self._strm_info_dict, f, indent=4)
    
def help_get_forecast_median(ids: list, date: str) -> list:
    import geoglows
    flows = []
    for id in ids:
        try:
            data: pd.DataFrame = geoglows.data.forecast_stats(id, date)
        except KeyError:
            msg = f"Could not get forecast for {id}; not in the geoglows dataset"
            LOG.warning(msg)
            gr.Warning(msg)
            continue
        
        flows.append(data['flow_med'].max())
        
    return flows

def open_csv_regardless_of_header(file_path: str) -> pd.DataFrame:
    # Check if the file has a header or not
    with open(file_path, 'r') as f:
        first_line = f.readline()
    if first_line[0].isdigit():
        header = None
    else:
        header = 0
    
    # Read the CSV file
    return pd.read_csv(file_path, header=header)
        

def get_selected_ids(ids_to_use: str, num_ids_to_use: int, num_upstream_branches: int, save_ids_file: str):
    try:
        import networkx
    except ImportError:
        msg = "Please install the networkx package to use this function. Run 'conda install networkx' in your terminal."
        LOG.error(msg)
        gr.Error(msg)
        return
    
    if not ids_to_use:
        msg = "No IDs to use specified"
        LOG.error(msg)
        gr.Error(msg)
        return
    
    if os.path.isfile(ids_to_use):
        if not os.path.exists(ids_to_use):
            msg = f"{ids_to_use} does not exist"
            LOG.error(msg)
            gr.Error(msg)
            return
        ids = open_csv_regardless_of_header(ids_to_use).iloc[:,0].values
    else:
        try:
            ids = [int(i) for i in ids_to_use.split(',')]
        except ValueError:
            msg = "You've entered invalid IDs"
            LOG.error(msg)
            gr.Error(msg)
            return

    # Create a networkx graph using streamlines
    