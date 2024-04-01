import gradio as gr
import os
import logging
from typing import Tuple, Dict, List
import pandas as pd
import geopandas as gpd
import json
import platform
import asyncio
import re
import glob
import sys

from osgeo import gdal, ogr
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
    def __init__(self, exe: str, mifn: str):
        self.manager = AutoRouteHandler(None)

    async def run(self):
        await self.run_exe(self.exe, self.mifn)

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
            cols = sorted(list(pd.read_csv(flow_file, delimiter='\t').columns))
            if len(cols) <= 1:
                try:
                    cols = sorted(list(pd.read_csv(flow_file, delimiter=' ').columns))
                except:
                    return None, None, None, None
        except:
            try:
                cols = sorted(list(pd.read_csv(flow_file, delimiter=' ').columns))
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

    def write_mifn(self,dem, mifn, strm, spatial_units, flow_file, subtract_baseflow, rowcols_from_flowfile, flow_id, flow_params, flow_baseflow, vdt, is_database, num_iterations,
                                                        meta_file, convert_cfs_to_cms, x_distance, q_limit, lu_raster, is_lu_same_as_dem, mannings_table, direction_distance, slope_distance, low_spot_distance, low_spot_is_meters,
                                                        low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, row_end, use_prev_d_4_xs,
                                                        weight_angles, man_n, adjust_flow, bathy_alpha, bathy_file, id_flow_file, omit_outliers, wse_search_dist, wse_threshold, wse_remove_three,
                                                        specify_depth, twd_factor, only_streams, use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, wse_map, fs_bathy_file,
                                                        da_flow_param,bathy_method,bathy_x_max_depth, bathy_y_shallow,fs_bathy_smooth_method, bathy_twd_factor) -> None:
        """
        Write the main input file
        """
        if not mifn:
            gr.Warning('Specify the main input file')
            return
        if not dem:
            gr.Warning('Specify the DEM')
            return
        
        # Format path strings
        mifn = self._format_files(mifn)
        dem = self._format_files(dem)
        
        if os.path.exists(mifn):
            gr.Warning(f'Overwriting main input file: {mifn}')
        
        if not isinstance(meta_file, bool) and os.path.exists(meta_file):
            gr.Warning(f'This will overwrite: {meta_file}')

        with open(mifn, 'w') as f:
            self._warn_DNE('DEM', dem)
            self._check_type('DEM', dem, ['.tif'])
            self._write(f,'DEM_File',dem)

            f.write('\n')
            self._write(f,'# AutoRoute Inputs')
            f.write('\n')

            if strm:
                strm = self._format_files(strm)
                self._warn_DNE('Stream File', strm)
                self._check_type('Stream Fil', strm, ['.tif'])
                self._write(f,'Stream_File',strm)

            self._write(f,'Spatial_Units',spatial_units)

            if flow_file:
                flow_file = self._format_files(flow_file)
                self._warn_DNE('Flow File', flow_file)
                self._check_type('Flow File', flow_file, ['.txt'])
                self._write(f,'Flow_RAPIDFile',flow_file)

                if rowcols_from_flowfile:
                    self._write(f,'RowCol_From_RAPIDFile')
                if not flow_id:
                    gr.Warning('Flow ID is not specified!!')
                else:
                    self._write(f,'RAPID_Flow_ID', flow_id)
                if not flow_params:
                    gr.Warning('Flow Params are not specified!!')
                else:
                    self._write(f,'RAPID_Flow_Param', " ".join(flow_params))
                if subtract_baseflow:
                    if not flow_baseflow:
                        gr.Warning('Base Flow Parameter is not specified, not subtracting baseflow')
                    else:
                        self._write(f,'RAPID_BaseFlow_Param',flow_baseflow)
                        self._write(f,'RAPID_Subtract_BaseFlow')

            if vdt:
                vdt = self._format_files(vdt)
                self._check_type('VDT', vdt, ['.txt'])
                if is_database:
                        self._write(f,'Print_VDT_Database',vdt)
                        self._write(f,'Print_VDT_Database_NumIterations',num_iterations)
                else:
                    self._write(f,'Print_VDT',vdt)

            if meta_file:
                meta_file = self._format_files(meta_file)
                self._write(f,'Meta_File',meta_file)

            if convert_cfs_to_cms: self._write(f,'CONVERT_Q_CFS_TO_CMS')

            self._write(f,'X_Section_Dist',x_distance)
            self._write(f,'Q_Limit',q_limit)
            self._write(f,'Gen_Dir_Dist',direction_distance)
            self._write(f,'Gen_Slope_Dist',slope_distance)
            self._write(f,'Weight_Angles',weight_angles)
            self._write(f,'Use_Prev_D_4_XS',use_prev_d_4_xs)
            self._write(f,'ADJUST_FLOW_BY_FRACTION',adjust_flow)
            if Str_Limit_Val: self._write(f,'Str_Limit_Val',Str_Limit_Val)
            if UP_Str_Limit_Val: self._write(f,'UP_Str_Limit_Val',UP_Str_Limit_Val)
            if row_start: self._write(f,'Layer_Row_Start',row_start)
            if row_end: self._write(f,'Layer_Row_End',row_end)

            if degree_manip > 0 and degree_interval > 0:
                self._write(f,'Degree_Manip',degree_manip)
                self._write(f,'Degree_Interval',degree_interval)

            if lu_raster:
                lu_raster = self._format_files(lu_raster)
                self._warn_DNE('Land Use', lu_raster)
                self._check_type('Land Use', lu_raster, ['.tif'])
                if is_lu_same_as_dem:
                    self._write(f,'LU_Raster_SameRes',lu_raster)
                else:
                    self._write(f,'LU_Raster',lu_raster)
                if not mannings_table:
                    gr.Warning('No mannings table for the Land Use raster!')
                else:
                    mannings_table = self._format_files(mannings_table)
                    self._write(f,'LU_Manning_n',mannings_table)
            else:
                self._write(f,'Man_n',man_n)

            if low_spot_distance:
                if low_spot_is_meters:
                    self._write(f,'Low_Spot_Dist_m',low_spot_distance)
                else:
                    self._write(f,'Low_Spot_Range',low_spot_distance)
                if low_spot_use_box:
                    self._write(f,'Low_Spot_Range_Box')
                    self._write(f,'Low_Spot_Range_Box_Size',box_size)
            
            if find_flat:
                if low_spot_find_flat_cutoff:
                    self._write(f,'Low_Spot_Find_Flat')
                    self._write(f,'Low_Spot_Range_FlowCutoff',low_spot_find_flat_cutoff)
                else:
                    gr.Warning('Low Spot Range cutoff was not defined')

            if bathy_file:
                bathy_file = self._format_files(bathy_file)
                self._write(f,'BATHY_Out_File',bathy_file)
                self._write(f,'Bathymetry_Alpha',bathy_alpha)

                if bathy_method == 'Parabolic':
                    self._write(f,'Bathymetry_Method',0)
                elif bathy_method == 'Left Bank Quadratic':
                    self._write(f,'Bathymetry_Method', 1)
                    self._write(f,'Bathymetry_XMaxDepth',bathy_x_max_depth)
                    self._write(f,'Bathymetry_YShallow',bathy_y_shallow)
                elif bathy_method == 'Right Bank Quadratic':
                    self._write(f,'Bathymetry_Method', 2)
                    self._write(f,'Bathymetry_XMaxDepth',bathy_x_max_depth)
                    self._write(f,'Bathymetry_YShallow',bathy_y_shallow)
                elif bathy_method == 'Double Quadratic':
                    self._write(f,'Bathymetry_Method', 3)
                    self._write(f,'Bathymetry_XMaxDepth',bathy_x_max_depth)
                    self._write(f,'Bathymetry_YShallow',bathy_y_shallow)
                elif bathy_method == 'Trapezoidal':
                    self._write(f,'Bathymetry_Method', 4)
                    self._write(f,'Bathymetry_XMaxDepth',bathy_x_max_depth)
                else: self._write(f,'Bathymetry_Method', 5)

                if da_flow_param: self._write(f, 'RAPID_DA_or_Flow_Param',da_flow_param)

            f.write('\n')
            self._write(f,'# FloodSpreader Inputs')
            f.write('\n')

            if id_flow_file:
                id_flow_file = self._format_files(id_flow_file)
                self._warn_DNE('ID Flow File', id_flow_file)
                self._check_type('ID Flow File', id_flow_file, ['.txt','.csv'])
                self._write(f,'Comid_Flow_File',id_flow_file)

            if omit_outliers == 'Flood Bad Cells':
                self._write(f,'Flood_BadCells')
            elif omit_outliers == 'Use AutoRoute Depths':
                self._write(f,'FloodSpreader_Use_AR_Depths')
            elif omit_outliers == 'Smooth Water Surface Elevation':
                self._write(f,'FloodSpreader_SmoothWSE_SearchDist',wse_search_dist)
                self._write(f,'FloodSpreader_SmoothWSE_FractStDev',wse_threshold)
                self._write(f,'FloodSpreader_SmoothWSE_RemoveHighThree',wse_remove_three)
            elif omit_outliers == 'Use AutoRoute Depths (StDev)':
                self._write(f,'FloodSpreader_Use_AR_Depths_StDev')
            elif omit_outliers == 'Specify Depth':
                self._write(f,'FloodSpreader_SpecifyDepth',specify_depth)

            self._write(f,'TopWidthDistanceFactor',twd_factor)
            if only_streams: self._write(f,'FloodSpreader_JustStrmDepths')
            if use_ar_top_widths: self._write(f,'FloodSpreader_Use_AR_TopWidth')
            if flood_local: self._write(f,'FloodLocalOnly')

            if depth_map:
                depth_map = self._format_files(depth_map)
                self._check_type('Depth Map',depth_map,['.tif'])
                self._write(f,'OutDEP',depth_map)

            if flood_map:
                flood_map =self._format_files(flood_map)
                self._check_type('Flood Map',flood_map,['.tif'])
                self._write(f,'OutFLD',flood_map)

            if velocity_map:
                velocity_map = self._format_files(velocity_map)
                self._check_type('Velocity Map',velocity_map,['.tif'])
                self._write(f,'OutVEL',velocity_map)

            if wse_map:
                wse_map = self._format_files(wse_map)
                self._check_type('WSE Map',wse_map,['.tif'])
                self._write(f,'OutWSE',wse_map)

            if fs_bathy_file and bathy_file: 
                fs_bathy_file = self._format_files(fs_bathy_file)
                self._check_type('FloodSpreader Generated Bathymetry',fs_bathy_file,['.tif'])
                self._write(f,'FSOutBATHY', fs_bathy_file)
                if fs_bathy_smooth_method == 'Linear Interpolation':
                    self._write(f,'Bathy_LinearInterpolation')
                elif fs_bathy_smooth_method == 'Inverse-Distance Weighted':
                    self._write(f,'BathyTopWidthDistanceFactor', bathy_twd_factor)
        
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