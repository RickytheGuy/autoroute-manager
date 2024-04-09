import gradio as gr
import os
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import contextily as ctx
import pandas as pd
import geopandas as gpd
import json
import platform
import asyncio
import re
import glob
import sys

from osgeo import gdal, ogr
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

docs_file = 'defaults_and_docs.json'
if not os.path.exists(docs_file):
    docs_file = os.path.join(os.path.dirname(__file__), 'defaults_and_docs.json')
if not os.path.exists(docs_file):
    gr.Warning('Could not find the defaults_and_docs.json file')
else:
    with open(docs_file, 'r', encoding='utf-8') as f:
        d = json.loads(f.read())[0]
        docs: Dict[str,str] = d['docs']

class ManagerFacade():
    def init(self) -> None:
        self.manager = AutoRouteHandler(None)
        
    async def run(self, **kwargs):
        await self._run(**kwargs)

    #@staticmethod
    def _format_files(self,file_path: str) -> str:
        """
        Function added so that windows paths that pad with quotation marks can be valid
        """
        if any(file_path.endswith(s) and file_path.startswith(s) for s in ['"',"'"]):
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
                gr.Warning(f"The file {value} for {card} does not exist: ")

    def _check_type(self, card: str, value: str, types: List[str]) -> None:
        if not any(value.endswith(t) for t in types):
            gr.Warning(f"{card} is {os.path.basename(value)}, which does not have a valid file type ({','.join(types)})")

    def omit_outliers_change(self,value: str) -> Tuple[gr.Radio, gr.Column, gr.Number]:
        if value == 'None':
            return gr.Radio(info='None: No outliers will be removed'), gr.Column(visible=False), gr.Number(visible=False)
        elif value == 'Flood Bad Cells':
            return gr.Radio(info=docs['Flood_BadCells']), gr.Column(visible=False), gr.Number(visible=False)
        elif value == 'Use AutoRoute Depths':
            return gr.Radio(info=docs['FloodSpreader_Use_AR_Depths']), gr.Column(visible=False), gr.Number(visible=False)
        elif value == 'Smooth Water Surface Elevation':
            return  gr.Radio(info=docs['smooth_wse']), gr.Column(visible=True), gr.Number(visible=False)
        elif value == 'Use AutoRoute Depths (StDev)':
            return gr.Radio(info=docs['FloodSpreader_Use_AR_Depths_StDev']), gr.Column(visible=False), gr.Number(visible=False)
        else:
            return gr.Radio(info=docs['FloodSpreader_SpecifyDepth']), gr.Column(visible=False), gr.Number(visible=True)

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
            cols = sorted(list(pd.read_csv(flow_file, delimiter=',').columns))
            if len(cols) <= 1:
                try:
                    cols = sorted(list(pd.read_csv(flow_file, delimiter=' ').columns))
                except:
                    return None, None, None, None
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

    async def _run(self,dem, dem_name, strm_lines, strm_name, lu_file, lu_name, base_max_file, subtract_baseflow, flow_id, flow_params, flow_baseflow, num_iterations,
                                                    meta_file, convert_cfs_to_cms, x_distance, q_limit, mannings_table, direction_distance, slope_distance, low_spot_distance, low_spot_is_meters,
                                                    low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, row_end, use_prev_d_4_xs,
                                                    weight_angles, man_n, adjust_flow, bathy_alpha, bathy_file, id_flow_file, omit_outliers, wse_search_dist, wse_threshold, wse_remove_three,
                                                    specify_depth, twd_factor, only_streams, use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, wse_map, fs_bathy_file, da_flow_param,
                                                    bathy_method,bathy_x_max_depth, bathy_y_shallow, fs_bathy_smooth_method, bathy_twd_factor,
                                                    data_dir, minx, miny, maxx, maxy, overwrite, buffer, crop, vdt_file) -> None:
        """
        Write the main input file
        """
        if minx == 0 and miny == 0 and maxx == 0 and maxy == 0:
            extent = None
        else:
            extent = (minx, miny, maxx, maxy)
        params = {"OVERWRITE": overwrite,
                  "DATA_DIR": data_dir,
                  "DEM_FOLDER": dem,
                  "BUFFER_FILES": buffer,
                  "DEM_NAME": dem_name,
                  "STREAM_NETWORK_FOLDER": strm_lines,
                  "STREAM_NAME": strm_name,
                  "STREAM_ID": flow_id,
                  "SIMULATION_FLOWFILE": base_max_file,
                  "ID_COLUMN": flow_id,
                  "FLOW_COLUMN": flow_params,
                  "BASE_FLOW_COLUMN": flow_baseflow,
                  "EXTENT": extent,
                  "CROP": crop,
                  "LAND_USE_FOLDER": lu_file,
                  "LAND_USE_NAME": lu_name,
                  "AUTOROUTE": "C:\Users\lrr43\Desktop\Lab\MichiganTest\AutoRoute_w_GDAL.exe", 
                  "FLOODSPREADER": "C:\Users\lrr43\Desktop\Lab\MichiganTest\AutoRoute_FloodSpreader.exe",
                  
}
        self.manager.setup(params)
        self.manager.run()

        
        
    async def run_exe(self,exe: str, mifn: str) -> None:
        """
        Run the executable, printing to terminal
        """
        if not exe or not mifn: return

        exe = self._format_files(exe.strip())
        mifn = self._format_files(mifn.strip())
        if not os.path.exists(exe):
            gr.Warning('AutoRoute Executable not found')
            return
        if not os.path.exists(mifn):
            gr.Warning('The Main Input File does not exist')
            return
        
        exe  = self._prepare_exe(exe)

        process = await asyncio.create_subprocess_shell(f'echo "a" | {exe} {mifn}', # We echo a dummy input in so that AutoRoute can terminate if some input is wrong
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
        while True:
            line = await process.stdout.readline()
            if not line: 
                logging.info('Program finished')
                break
            logging.info(line.decode().strip())

        await process.communicate()
        
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
        ctx.add_basemap(ax, 
                    crs=3857, 
                    attribution=False, 
                    source=ctx.providers.Esri.WorldImagery
                    )
        
        return fig
        