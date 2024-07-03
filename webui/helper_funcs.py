import gradio as gr
import os
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import geopandas as gpd
import json
import platform
import asyncio
import re
import glob
import sys
import multiprocessing
import xarray as xr
import numpy as np

from osgeo import gdal, ogr
from shapely.geometry import box
from git import Repo
from shapely.geometry import LineString
from pyproj import Transformer
try:
    from autoroute.autoroute import AutoRouteHandler
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    from autoroute.autoroute import AutoRouteHandler

gdal.UseExceptions()

SYSTEM = platform.system()

class ManagerFacade():
    def init(self) -> None:
        self.manager = AutoRouteHandler(None)
        self.docs_file = 'defaults_and_docs.json'
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
        
        extensions = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extensions')
        self.pull_autoroute_py(extensions)
        from extensions.autoroutepy import Automated_Rating_Curve_Generator
        self.autoroutepy = Automated_Rating_Curve_Generator.main

        
    async def run(self, **kwargs):
        await self._run(**kwargs)

    def doc(self, key: str) -> str:
        return self.docs.get(key, 'No documentation available')
    
    def default(self, key: str) -> str:
        return self.data.get(key, None)
    
    def get_ids(self, *args) -> None:
        logging.info("Getting IDs...")
        dem = args[0]
        strm_lines = self._format_files(args[1])
        minx = args[2]
        miny = args[3]
        maxx = args[4]
        maxy = args[5]
        ids_folder = self._format_files(args[6])
        flow_id = args[7]
        
        if os.path.isdir(strm_lines):
            stream_files = [os.path.join(strm_lines,f) for f in os.listdir(strm_lines) if f.endswith(('.shp', '.gpkg', '.parquet', '.geoparquet'))]
        elif os.path.isfile(strm_lines):
            stream_files = [strm_lines]
        else:
            msg = f"{strm_lines} is not a valid file or folder"
            logging.error(msg)
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
            except ValueError:
                msg = f"{flow_id !r} is not a valid field in the stream files provided"
                logging.error(msg)
                gr.Error(msg)
                return
        if len(results) == 0:
            msg = "No IDs found in the stream files provided"
            logging.error(msg)
            gr.Error(msg)
            return
        if len(results) == 1:
            results = np.unique(results[0])
        else:
            results = np.unique(np.concatenate(results, axis=None))
        pd.DataFrame({flow_id: results}).to_csv(output_file, index=False)
        msg = f"IDs saved to {output_file}"
        logging.info(msg)
        gr.Info(msg)
        
        
    def _get_ids(self, strm_lines: str, extent: List[float], id_field: str) -> np.ndarray:
        bbox = box(extent[0], extent[1], extent[2], extent[3])
        gdf = self.manager.gpd_read(strm_lines, columns=[id_field], bbox=bbox)
        return gdf[id_field].to_numpy().astype(int)
        
    
    def save(self, *args) -> None:
        """
        Save the modified documentation and defaults

        order of args: 
        0. dem, dem_name, strm_lines, strm_name, lu_file, 
        5. lu_name, base_max_file, subtract_baseflow, flow_id, flow_params, 
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
        if len(args) < 69: return
        to_write = [
            {
                "docs": self.docs,
                "data": {
                    "DEM":args[0],
                    "DEM_NAME":args[1],
                    "strm_lines":args[2],
                    "strm_name":args[3],
                    "flow_id":args[8],
                    "ar_exe":args[67],
                    "fs_exe":args[68],
                    "x_distance":args[14],
                    "q_limit":args[15],
                    "use_prev_d_4_xs":args[31],
                    "man_n":args[33],
                    "lu_file":args[4],
                    "lu_name":args[5],
                    "LU_Manning_n":args[16],
                    "base_max_file":args[6],
                    "Gen_Dir_Dist":args[17],
                    "Gen_Slope_Dist":args[18],
                    "Weight_Angles":args[32],
                    "Str_Limit_Val":args[27],
                    "UP_Str_Limit_Val":args[28],
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
                    "Layer_Row_Start":args[29],
                    "Layer_Row_End":args[30],
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
            
        logging.info("Parameters saved!")

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

    def bathy_changes(self,value: str) -> Tuple[gr.Slider, gr.Slider]:
        if value == 'Trapezoidal':
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

    async def _run(self,dem, dem_name, strm_lines, strm_name, lu_file, lu_name, base_max_file, subtract_baseflow, flow_id, flow_params_ar, flow_baseflow, num_iterations,
                                                    meta_file, convert_cfs_to_cms, x_distance, q_limit, mannings_table, direction_distance, slope_distance, low_spot_distance, low_spot_is_meters,
                                                    low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, row_end, use_prev_d_4_xs,
                                                    weight_angles, man_n, adjust_flow, bathy_alpha, ar_bathy, id_flow_file, omit_outliers, wse_search_dist, wse_threshold, wse_remove_three,
                                                    specify_depth, twd_factor, only_streams, use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, wse_map, fs_bathy_file, da_flow_param,
                                                    bathy_method,bathy_x_max_depth, bathy_y_shallow, fs_bathy_smooth_method, bathy_twd_factor,
                                                    data_dir, minx, miny, maxx, maxy, overwrite, buffer, crop, vdt_file,  ar_exe, fs_exe, clean_outputs, buffer_distance, use_ar_python, run_ar_bathy) -> None:
        """
        Write the main input file
        """
        if {minx, miny, maxx, maxy} == {0}:
            extent = None
        else:
            extent = (minx, miny, maxx, maxy)
        params = {"OVERWRITE": overwrite,
                  "DATA_DIR": self._format_files(data_dir),
                  "DEM_FOLDER": self._format_files(dem),
                  "BUFFER_FILES": buffer,
                  "BUFFER_DISTANCE": buffer_distance,
                  "DEM_NAME": dem_name,
                  "STREAM_NETWORK_FOLDER": self._format_files(strm_lines),
                  "STREAM_NAME": strm_name,
                  "STREAM_ID": flow_id,
                  "SIMULATION_FLOWFILE": self._format_files(base_max_file),
                  "FLOOD_FLOWFILE":self._format_files(id_flow_file),
                  "SIMULATION_ID_COLUMN": flow_id,
                  "SIMULATION_FLOW_COLUMN": flow_params_ar,
                  "BASE_FLOW_COLUMN": flow_baseflow,
                  "EXTENT": extent,
                  "CROP": crop,
                  "LAND_USE_FOLDER": self._format_files(lu_file),
                  "LAND_USE_NAME": lu_name,
                  "MANNINGS_TABLE": self._format_files(mannings_table),
                  "DEPTH_MAP": self._format_files(depth_map),
                  "FLOOD_MAP": self._format_files(flood_map),
                  "VELOCITY_MAP": self._format_files(velocity_map),
                  "WSE_MAP": self._format_files(wse_map),
                  "CLEAN_OUTPUTS": clean_outputs,
                  
                  "AUTOROUTE_PYTHON_MAIN": self.autoroutepy,
                  "AUTOROUTE": self._format_files(ar_exe),
                  "FLOODSPREADER": self._format_files(fs_exe),
                  "AUTOROUTE_CONDA_ENV": "autoroute",

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
                  "Str_Limit_Val": Str_Limit_Val,
                  "UP_Str_Limit_Val": UP_Str_Limit_Val,
                  "row_start": row_start,
                  "row_end": row_end,
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
    
    def make_map(self, 
                 minx: float, 
                 miny: float, 
                 maxx: float, 
                 maxy: float):
        if minx is None:
            minx = -180
        if miny is None:
            miny = -90
        if maxx is None:
            maxx = 180
        if maxy is None:
            maxy = 90

        if {minx, miny, maxx, maxy} == {0}: # If all are 0, then we have no data
            gr.Info("0 means no extent was specified")
            return

        if maxx <= minx:
            gr.Warning('Max X is less than Min X')
            return
        if maxy <= miny:
            gr.Warning('Max Y is less than Min Y')
            return
        if abs(miny) > 90  or abs(maxy) > 90 or maxx > 180 or minx < -180:
            gr.Warning('Invalid coordinates')
            return
        transformer = Transformer.from_crs(f"EPSG:4326","EPSG:3857", always_xy=True) 
        minx, miny = transformer.transform(minx, miny)
        maxx, maxy =  transformer.transform(maxx, maxy)
  
        dif = (((maxx - minx) + (maxy - miny)) / 2 ) * 0.25
        
        # Make some geometry to plot
        df = gpd.GeoDataFrame(geometry=[LineString([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])],crs=3857)
        
        
        plt.rcParams['figure.figsize'] = [12, 6]
        fig = plt.figure(edgecolor='black')
        ax = fig.add_axes([0, 0, 1, 1])
        df.plot(ax=ax)
        ax.set_xlim(max(minx - dif, -20026376.39), min(maxx + dif, 20026376.39))
        ax.set_ylim(max(miny - dif, -20048966.1), min(maxy + dif, 20048966.1))
        ax.set_xticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_yticks([])
        ax.tick_params(left = False, bottom = False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        try:
            import contextily as ctx
            ctx.add_basemap(ax, 
                        crs=3857, 
                        attribution=False, 
                        source=ctx.providers.Esri.WorldImagery
                        )
        except ImportError:
            pass
        
        return fig
        
    def pull_autoroute_py(self,extensions: str) -> None:
        """
        Pull the autoroute.py file from the extensions folder
        """
        if not os.path.exists(extensions):
            os.makedirs(extensions)
            
        import_folder = os.path.join(extensions, 'autoroutepy')
        if not os.path.exists(import_folder):
            try:
                Repo.clone_from('https://github.com/RickytheGuy/automated-rating-curve-byu.git', import_folder)
            except Exception as e:
                gr.Warning(f"Could not clone the automated rating curve repository: {e}")
            return
                
        repo = Repo(import_folder)
        try:
            repo.remotes.origin.pull()
        except Exception as e:
            gr.Warning(f"Could not pull the automated rating curve repository: {e}")
            
                