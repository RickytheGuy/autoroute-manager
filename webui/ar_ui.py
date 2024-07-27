"""
UI designed to easily use AutoRoute and FloodSpreader. Tools to create input files. Installation helps.

Louis "Ricky" Rosas
BYU HydroInformatics Lab
"""
import gradio as gr
import sys
import signal
import logging

import helper_funcs as hp
import datetime
manager = hp.ManagerFacade()

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def shutdown(signum, server):
    """
    When closing python script, shutdown the server
    """
    print()
    print('Shutting down...')
    demo.close()
    exit()

signal.signal(signal.SIGINT, shutdown) # Control 
if not hp.SYSTEM == 'Windows':
    signal.signal(signal.SIGTSTP, shutdown)

if __name__ == '__main__':
    manager.init()

    with gr.Blocks(title='AutoRoute WebUI') as demo:
        gr.Markdown('# AutoRoute WebUI')
            
        with gr.Tabs():
            with gr.TabItem('Run AutoRoute'):
                gr.Markdown('## Inputs - Required')
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            dem = gr.Textbox(value=manager.default('DEM'),
                                            placeholder='/User/Desktop/dem.tif',
                                            label="Digital Elevation Model (DEM) Folder or File",
                                            #info=manager.default(\w+),
                                            )
                            dem_name = gr.Textbox(value=manager.default('DEM_NAME'),
                                                placeholder='Copernicus',
                                                label='DEM Name',
                                                visible=False
                                                #info=manager.default(1),
                                                )
                        with gr.Row():
                            strm_lines = gr.Textbox(value=manager.default("strm_lines"), 
                                                    placeholder='/User/Desktop/dem.tif',
                                                    label="Stream Lines Folder or File",
                                                    #info=manager.default('DEM'),
                                                    )
                            strm_name = gr.Textbox(value=manager.default("strm_name"), 
                                                placeholder='Copernicus',
                                                label='Streamlines Name',
                                                visible=False,
                                                #info=manager.default('DEM_NAME'),
                                                )
                            streamlines_id_col = gr.Dropdown(value=manager.default("flow_id"),
                                                label='Streamlines ID Column',
                                                #info='Specifies the stream identifier that AutoRoute uses. Leave blank to use the first column.',
                                                allow_custom_value=True,
                                                multiselect=False,
                                                interactive=True
                                                )
                        
                        with gr.Row():
                            lu_file = gr.Textbox(value=manager.default("lu_file"), 
                                                placeholder='/User/Desktop/lu.tif',
                                                label="Land Raster Folder or File",
                                                #info=manager.default('DEM'),
                                                )
                            lu_name = gr.Textbox(value=manager.default("lu_name"),
                                                placeholder='Copernicus',
                                                label='Land Raster Name',
                                                #info=manager.default('DEM_NAME'),
                                                visible=False
                                                )
                            LU_Manning_n = gr.Textbox(value=manager.default("LU_Manning_n"),
                                                    placeholder='/User/Desktop/mannings_n.txt',
                                                    label="Manning's n table",
                                                    #info=manager.default('LU_Manning_n')
                            )
                           
                        id_flow_file = gr.Textbox(value=manager.default("Comid_Flow_File"),
                                        placeholder='/User/Desktop/100_year_flow.txt',
                                        label="ID Flow File",
                                        #info=manager.default('Comid_Flow_File')
                            )

                        base_max_file = gr.Textbox(value=manager.default("base_max_file"),
                                                placeholder='/User/Desktop/flow_file.txt',
                                                label="Base and Max Flow File",
                                                #info=manager.default('Flow_RAPIDFile'),
                        )
                        with gr.Row():
                            
                            flow_params_ar = gr.Dropdown(value=manager.default("flow_params_ar"),
                                                    label='Max Flow Column',
                                                    #info='Specifies the flow rates that AutoRoute uses. Leave blank to use all columns besides the first one.',
                                                    allow_custom_value=True,
                                                    multiselect=False,
                                                    interactive=True) # Careful
                            flow_baseflow = gr.Dropdown(value=manager.default("flow_baseflow"),label='Base Flow Column',
                                                    #info='Specifies the base flow rates that AutoRoute uses. Leave blank to not use.',
                                                    allow_custom_value=True,
                                                    multiselect=False,
                                                    interactive=True)
                            subtract_baseflow = gr.Checkbox(value=manager.default("subtract_baseflow"),
                                                            label='Subtract Base Flow?',
                                                            interactive=True
                            )
                                
                        vdt_file = gr.Textbox(value=manager.default("vdt"),
                                    placeholder='/User/Desktop/VDT/',
                                    label="VDT Folder (Optional)",
                                    info=manager.doc('vdt')
                        )

                        with gr.Row():
                            crop = gr.Checkbox(value=manager.default("crop"),
                                            label='Crop',
                                                info='Crop output to extent?',
                                                interactive=True)
                            clean_outputs = gr.Checkbox(value=manager.default("clean_outputs"),
                                                label='Optimize Outputs',
                                                info='Trim output raster file sizes, add compression',
                                                interactive=True)
                            overwrite = gr.Checkbox(False, visible=True,
                                                    label='Overwrite',
                                                interactive=True,
                                                info="Force overwrite of existing files. Will take the longest time.")
                            with gr.Column():
                                buffer = gr.Checkbox(value=manager.default("buffer"),
                                                        visible=False,
                                                    label='Buffer',
                                                    info='Buffer the DEMs?',
                                                    interactive=True)
                                buffer_distance = gr.Number(value=manager.default("buffer_distance"),
                                                            label='Buffer Distance',
                                                            visible=manager.default("buffer"),
                                                            interactive=True)
                                buffer.change(lambda x: gr.Number(visible=x), inputs=buffer, outputs=buffer_distance)
                        
                    with gr.Column(scale=2):
                        map_output = gr.Plot(label="Extent Preview")
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    gr.Markdown("Specify an extent if needed")
                                    minx = gr.Number(value=manager.default("minx"),
                                                    label='Min X', )
                                    maxx = gr.Number(value=manager.default("maxx"),
                                                    label='Max X', )
                                    miny = gr.Number(value=manager.default("miny"),
                                                    label='Min Y', )
                                    maxy = gr.Number(value=manager.default("maxy"),
                                                    label='Max Y', )

                                map_button = gr.Button("Preview Extent on Map")
                                map_button.click(fn=manager.make_map, inputs=[minx, miny, maxx, maxy], outputs=[map_output])

                                data_dir = gr.Textbox(value=manager.default("data_dir"),
                                                    label='Data Directory',
                                                    info='Directory where AutoRoute will store its data',
                                                    interactive=True)
                                
                                use_ar_python = gr.Checkbox(value=manager.default("use_ar_python"),
                                                        label='Use Automated Rating Curve',
                                                        info='Use Python for AutoRoute?',
                                                        interactive=True)
                                
                                ar_exe = gr.Textbox(value=manager.default("ar_exe"),
                                                    placeholder='/User/Desktop/AutoRoute.exe',
                                                    label="AutoRoute Executable",
                                                    info=manager.doc('ar_exe'),
                                                    visible=True
                                )
                                use_ar_python.change(lambda x: gr.Textbox(visible=not x), inputs=use_ar_python, outputs=ar_exe)
                                

                                fs_exe = gr.Textbox(value=manager.default("fs_exe"),
                                                    placeholder='/User/Desktop/FloodSpreader.exe',
                                                    label="FloodSpreader Executable",
                                                    info=manager.doc('fs_exe')
                                )
                                ids_folder = gr.Textbox(value='',
                                                        placeholder='/User/Desktop/ids.txt',
                                                        label='IDs Folder (optional)',
                                                        info="Folder to save IDs found in the given extent. Leave blank to not save")

                            with gr.Column():
                                depth_map = gr.Textbox(value=manager.default("out_depth"),
                                    placeholder='/User/Desktop/Depth/',
                                    label="Output Depth Map Folder",
                                    #info=manager.doc('out_depth')
                                )
                                flood_map = gr.Textbox(value=manager.default("out_flood"),
                                    placeholder='/User/Desktop/flood/',
                                    label="Output Flood Map Folder",
                                    #info=manager.doc('out_flood')
                                )
                                velocity_map = gr.Textbox(value=manager.default("out_velocity"),
                                    placeholder='/User/Desktop/velocity',
                                    label="Output Velocity Map Folder",
                                    #info=manager.doc('out_velocity')
                                )
                                wse_map = gr.Textbox(value=manager.default("out_wse"),
                                    placeholder='/User/Desktop/wse',
                                    label="Output WSE Map Folder",
                                    #info=manager.doc('out_wse')
                                )

                                run_button = gr.Button("Run Model", variant='primary')
                                save_button = gr.Button("Save Parameters")
                                get_ids_button = gr.Button("Get IDs from Inputs Given", variant='secondary')
                                
                            # with gr.Column():
                            #     gr.Markdown('Below are the options to select a river and its downstream neighbors to use. '
                            #                 '')
                                
                            #     ids_to_use = gr.Textbox(placeholder='/User/Desktop/ids.txt or 123456',
                            #                             label='IDs to Use',
                            #                             info=manager.doc('ids_to_use'))
                            #     num_ids_to_use = gr.Number(value=-1, label='Number of IDs to Use', info=("This specifies how many downstream ids to include besides the ID specified"
                            #                                                                              " Does nothing is a file is provided above. -1 means all downstream IDs are included"))
                            #     num_upstream_branches = gr.Number(value=0, label='Number of Branches Upstream to Include', 
                            #                                       info=("This specifies how many upstream branches to include"
                            #                                     " for the downstream ids. 0 means only the downstream ids are included. Does nothing if a file is provided above"))
                            #     save_ids_file = gr.Textbox(placeholder='/User/Desktop/ids.txt',
                            #                             label='Save IDs to File',
                            #                             info="Save the selected ids above to a file to use later. Leave blank to not save")
                            #     get_selected_ids = gr.Button("Get Selected IDs")
                            #     get_selected_ids.click(fn=manager.get_selected_ids, inputs=[ids_to_use, num_ids_to_use, num_upstream_branches, save_ids_file], outputs=[])
                      
                gr.Markdown('## Inputs - Optional')  
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("AutoRoute parameters"):
                            adjust_flow = gr.Number(value=manager.default("ADJUST_FLOW_BY_FRACTION"),
                                            label='Adjust Flow',
                                            info=manager.doc('ADJUST_FLOW_BY_FRACTION'),
                                            interactive=True)
                        
                            num_iterations = gr.Number(value=manager.default("num_iterations"),
                                                minimum=1,
                                                label='VDT Database Iterations',
                                                info=manager.doc('num_iterations'),
                                                interactive=True,)
                
                            meta_file = gr.Textbox(value=manager.default("Meta_File"),
                                            placeholder='/User/Desktop/meta.txt',
                                            label="Meta File",
                                            info=manager.doc('Meta_File')
                                )

                            with gr.Row():
                                convert_cfs_to_cms = gr.Checkbox(value=manager.default("convert_cfs_to_cms"),
                                                                label='CFS to CMS (THIS ONLY WORKS WITHOUT BASE MAX FILE)',
                                                                info='Convert flow values from cubic feet per second to cubic meters per second'
                                )

                            with gr.Row():
                                x_distance = gr.Slider(0,
                                                    50_000,
                                                    value=manager.default("x_distance"),
                                                    step=1,
                                                    label='Cross Section Distance',
                                                    info=manager.doc('x_distance'),
                                                    interactive=True
                                                    )
                                q_limit = gr.Slider(0,
                                                    2,
                                                    value=manager.default("q_limit"),
                                                    label='Flow Limit',
                                                    info=manager.doc('q_limit'),
                                                    interactive=True)
                            

                            with gr.Row():
                                direction_distance = gr.Slider(1,500,
                                                            value=manager.default("Gen_Dir_Dist"),
                                                            step=1,
                                                            label='Direction Distance',
                                                            info=manager.doc('Gen_Dir_Dist'),
                                                            interactive=True)
                                
                                slope_distance = gr.Slider(1,
                                                        500,
                                                        value=manager.default("Gen_Slope_Dist"),
                                                        step=1,
                                                        label='Slope Distance',
                                                            info=manager.doc('Gen_Slope_Dist'),
                                                            interactive=True)
                                
                            with gr.Row():
                                low_spot_distance = gr.Slider(0,500,value=manager.default("Low_Spot_Range"),
                                                    step=1,
                                                    label='Low Spot Distance',
                                                    info=manager.doc('Low_Spot_Range'),
                                                    interactive=True)
                                with gr.Column():
                                    low_spot_is_meters = gr.Checkbox(value=manager.default("low_spot_is_meters"),
                                                                    label='Is Meters?')
                                    low_spot_use_box = gr.Checkbox(value=manager.default("low_spot_use_box"),
                                                                label='Use a Range Box?')
                                    box_size = gr.Slider(1,10,value=manager.default("box_size"),
                                                        step=1,
                                                        label='Box Size',
                                                        visible=manager.default("low_spot_use_box"),
                                                        interactive=True)
                                    low_spot_use_box.change(lambda x: gr.Slider(visible=x), inputs=low_spot_use_box, outputs=box_size)

                                    find_flat = gr.Checkbox(value=manager.default("find_flat"), label='Find Flat?')
                                    low_spot_find_flat_cutoff = gr.Number(value=manager.default("Low_Spot_Find_Flat"),
                                                                        label='Flow Cutoff',
                                                                        info='Low_Spot_Find_Flat',
                                                                        visible=manager.default("find_flat"),
                                                                        interactive=True
                                                                        )
                                    find_flat.change(lambda x: gr.Number(visible=x), inputs=find_flat, outputs=low_spot_find_flat_cutoff)

                            with gr.Accordion('Sample Additional Cross-Sections', open=False):
                                gr.Markdown(manager.doc('degree'))
                                with gr.Row():
                                    degree_manip = gr.Number(value=manager.default("degree_manip"), label='Farthest Angle Out (Degree_Manip)')
                                    degree_interval = gr.Number(value=manager.default("degree_interval"), label='Angle Between Cross-Sections (Degree_Interval)')
                                    
                            with gr.Accordion('Set Bounds on Stream Raster', open=False):
                                gr.Markdown(manager.doc('limit_vals'))
                                with gr.Row():
                                    Str_Limit_Val = gr.Number(value=manager.default("Str_Limit_Val"), label='Lowest Perissible Value')
                                    UP_Str_Limit_Val = gr.Number(value=manager.default("UP_Str_Limit_Val"), label='Highest Perissible Value')
                            
                            with gr.Row():
                                # DEPRECATED
                                row_start=gr.Number(value=manager.default("Layer_Row_Start"),
                                                    precision=0,
                                                    label='Starting Row',
                                                    info=manager.doc('Layer_Row_Start'),
                                                    visible=False)
                                row_end=gr.Number(value=manager.default("Layer_Row_End"),
                                                precision=0,
                                                    label='End Row',
                                                    info=manager.doc('Layer_Row_End'),
                                                    visible=False)
                                    
                            with gr.Row():     
                                use_prev_d_4_xs = gr.Dropdown(
                                    [0,1],
                                    value=manager.default("use_prev_d_4_xs"),
                                    label='Use Previous Depth for Cross Section',
                                    info=manager.doc('use_prev_d_4_xs'),
                                    interactive=True
                                )

                                weight_angles = gr.Number(value=manager.default("Weight_Angles"),
                                                label='Weight Angles',
                                                info=manager.doc('Weight_Angles'),
                                                interactive=True,
                                                )

                                man_n = gr.Number(value=manager.default("man_n"),
                                                label='Manning\'s n Value',
                                                info=manager.doc('man_n'),
                                                interactive=True,
                                                )
                                
                            lu_name.change(manager.show_mans_n, [lu_name,LU_Manning_n], man_n)

                            with gr.Accordion('Bathymetry'):
                                run_ar_bathy = gr.Checkbox(value=manager.default("run_ar_bathy"),
                                                            interactive=True,
                                                            label='Run AutoRoute Bathymetry?',
                                                                   )
                                        
                                bathy_row = gr.Row(visible=False)
                                with bathy_row:
                                    with gr.Column():
                                        
                                        ar_bathy_out_file = gr.Textbox(value=manager.default("BATHY_Out_File"),
                                                    placeholder='/User/Desktop/bathy/',
                                                    label="Output Bathymetry File",
                                                    #info=manager.doc('BATHY_Out_File')
                                                    info="Output folder for AutoRoute bathymetry. Does not need to be specified for bathymetry to run"
                                        )
                                        bathy_alpha = gr.Number(value=manager.default("Bathymetry_Alpha"),
                                                                label='Bathymetry Alpha',
                                                                info=manager.doc('Bathymetry_Alpha'),
                                                                interactive=True,
                                                                )
                                        da_flow_param = gr.Dropdown(value=manager.default("RAPID_DA_or_Flow_Param"),
                                                                    label='Drainage or Flow Parameter',
                                                                info=manager.doc('RAPID_DA_or_Flow_Param'),
                                                                allow_custom_value=True,
                                                                multiselect=False,interactive=True)

                                    with gr.Column():
                                        bathy_method = gr.Dropdown(['Parabolic', 'Left Bank Quadratic', 'Right Bank Quadratic', 'Double Quadratic', 'Trapezoidal','Triangle'],
                                                                value=manager.default("bathy_method"),
                                                                label='Bathymetry Method',
                                                                info=manager.doc('bathy_method'),
                                                                multiselect=False,
                                                                interactive=True,
                                                                allow_custom_value=True)
                                        bathy_x_max_depth = gr.Slider(0,1,value=manager.default("bathy_x_max_depth"),
                                                                    label='X Max Depth',
                                                                    info=manager.doc('bathy_x_max_depth'), 
                                                                    visible=False)
                                        bathy_y_shallow = gr.Slider(0,1,value=manager.default("bathy_y_shallow"),
                                                                    label='Y Shallow',
                                                                    info=manager.doc('bathy_y_shallow'), 
                                                                    visible=False)
                                        
                                        bathy_method.change(manager.bathy_changes, bathy_method, [bathy_x_max_depth, bathy_y_shallow])
                                
                                run_ar_bathy.change(lambda x: gr.Row(visible=x), inputs=run_ar_bathy, outputs=bathy_row)
                                base_max_file.change(manager.update_flow_params, base_max_file, [streamlines_id_col,flow_params_ar, flow_baseflow, da_flow_param])

                    with gr.Column():
                        with gr.Accordion("FloodSpreader Parameters"):
                            with gr.Column():
                                omit_outliers = gr.Radio(['None','Flood Bad Cells', 'Use AutoRoute Depths', 'Smooth Water Surface Elevation','Use AutoRoute Depths (StDev)','Specify Depth'],
                                        value=manager.default("omit_outliers"),
                                        label='Omit Outliers',
                                        interactive=True,
                                        info='None: No outliers will be removed'
                                        )
                                
                                wse_col = gr.Column(visible=False)
                                with wse_col:
                                    with gr.Row():
                                        wse_search_dist = gr.Slider(1,100,value=manager.default("wse_search_dist"),
                                                step=1,
                                                label='Smooth WSE Search Distance',
                                                info=manager.doc('wse_search_dist'),
                                                interactive=True)
                                        wse_threshold = gr.Number(value=manager.default("wse_threshold"),
                                                                label='Smooth WSE Threshold',
                                                                info=manager.doc('wse_threshold'),
                                                                interactive=True)
                                        wse_remove_three = gr.Checkbox(value=manager.default("wse_remove_three"),
                                                                    label='Smooth WSE Remove Highest Three',
                                                                    info=manager.doc('wse_remove_three'),
                                                                    interactive=True)
                                        
                                specify_depth = gr.Number(value=manager.default("FloodSpreader_SpecifyDepth"),
                                                        label='Specify Depth',
                                                            interactive=True,
                                                            visible='Specify Depth' in manager.default("omit_outliers"))
                                omit_outliers.change(manager.omit_outliers_change, inputs=omit_outliers, outputs=[omit_outliers, wse_col, specify_depth])

                                with gr.Row():
                                    twd_factor = gr.Slider(0,10,value=manager.default("twd_factor"),
                                                        label='Top Width Distance Factor',
                                                        info=manager.doc('twd_factor'),
                                                        interactive=True)
                                    with gr.Column():
                                        only_streams = gr.Checkbox(value=manager.default("only_streams"),
                                                                label='Only Output Values for Stream Locations',
                                                                info=manager.doc('only_streams'),
                                                                interactive=True)
                                        use_ar_top_widths = gr.Checkbox(value=manager.default("use_ar_top_widths"),
                                                                label='Use AutoRoute Top Widths',
                                                                info=manager.doc('use_ar_top_widths'),
                                                                interactive=True)
                                        flood_local = gr.Checkbox(value=manager.default("FloodLocalOnly"),
                                                                label='Flood Local',
                                                                info=manager.doc('FloodLocalOnly'),
                                                                interactive=True)
                            

                            with gr.Accordion('Bathymetry'):
                                gr.Markdown('Note that the bathymetry file generated by AutoRoute must be specified in the AutoRoute Bathymetry section')
                                with gr.Row():
                                    fs_bathy_file = gr.Textbox(value=manager.default("fs_bathy_file"),
                                                placeholder='/User/Desktop/floodspreader_bathy/',
                                                label="DEM with Bathymetry Burned In",
                                                #info=manager.doc('fs_bathy_file')
                                                info="Output folder for bathymetry-burned DEMs. MUST be specified"
                                    )
                                    with gr.Column():
                                        fs_bathy_smooth_method = gr.Dropdown(['None','Linear Interpolation', 'Inverse-Distance Weighted'],
                                                                            value=manager.default("fs_bathy_smooth_method"),
                                                                            label='Bathymetry Smoothing',
                                                                            interactive=True) 
                                        bathy_twd_factor = gr.Number(value=manager.default("bathy_twd_factor"),
                                                                    label='Bathymetry Top Width Distance Factor',
                                                                    interactive=True,
                                                                    visible='Inverse-Distance Weighted' in manager.default("fs_bathy_smooth_method"))
                                        fs_bathy_smooth_method.change(lambda x: gr.Number(visible=True) if x[0] == 'I' else gr.Number(visible=False),
                                                                    fs_bathy_smooth_method, bathy_twd_factor)
                
                inputs = [dem,dem_name, strm_lines, strm_name, lu_file, lu_name, base_max_file, subtract_baseflow, streamlines_id_col, flow_params_ar, flow_baseflow, num_iterations,
                                                        meta_file, convert_cfs_to_cms, x_distance, q_limit, LU_Manning_n, direction_distance, slope_distance, low_spot_distance, low_spot_is_meters,
                                                        low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, row_end, use_prev_d_4_xs,
                                                        weight_angles, man_n, adjust_flow, bathy_alpha, ar_bathy_out_file, id_flow_file, omit_outliers, wse_search_dist, wse_threshold, wse_remove_three,
                                                        specify_depth, twd_factor, only_streams, use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, wse_map, fs_bathy_file, da_flow_param,
                                                        bathy_method,bathy_x_max_depth, bathy_y_shallow, fs_bathy_smooth_method, bathy_twd_factor,
                                                        data_dir, minx, miny, maxx, maxy, overwrite, buffer, crop, vdt_file, ar_exe, fs_exe, clean_outputs, buffer_distance,
                                                        use_ar_python, run_ar_bathy]
                try:
                    run_button.click(fn=manager._run, inputs=inputs, outputs=[])
                except Exception as e:
                    gr.Error(e)
                    logging.error(e)
                                                
                save_button.click(fn=manager.save, inputs=inputs, outputs=[])
                get_ids_button.click(fn=manager.get_ids, inputs=[dem, strm_lines, minx, miny, maxx, maxy, ids_folder, streamlines_id_col], outputs=[])

            with gr.TabItem('File Preprocessing'):
                with gr.Tabs():
                    with gr.TabItem("Get Forecast Data for GEOGLoWS IDs"):
                        gr.Markdown("## Get Forecast Data for GEOGLoWS IDs\n\nThis tool will get forecast data for a list of GEOGLoWS IDs. The IDs must be a file with only one column. A header may or may not be present. This function will get the data from geoglows V2, select just the enesmble, and get the maximum values from that ensemble across all of it's time. The output file is guarentted to have two columns, one named 'LINKNO' and the other 'max_forecast'.")
                        id_list  = gr.Textbox(value=manager.default("id_forecast_list"), 
                                                            placeholder='/User/Desktop/id_list.csv',
                                                            label="ID File",
                                                            info=manager.doc('id_forecast_list'),
                                                            )
                        
                        date = gr.Textbox(datetime.datetime.now().strftime("%Y%m%d"),
                                            label='Date',
                                            info='Date for forecast data. Must be entered in YYYYMMDD format',
                                            )
                        
                        ensemble = gr.Textbox(value='ensemble_52',
                                              label='Ensemble',
                                            info='Ensemble for forecast data',
                                            )
                        
                        output_forecast_file = gr.Textbox(placeholder='/User/Desktop/forecast.csv',
                                                        label='Output File',
                                                        info='Output file for forecast data',
                                                        )
                        
                        get_forecast_button = gr.Button("Get Forecast Data")
                        get_forecast_button.click(fn=manager.get_forecast, inputs=[id_list, output_forecast_file, date, ensemble], outputs=[])
                    
                    with gr.TabItem("Get Median Forecast Data for GEOGLoWS IDs"):
                        gr.Markdown("## Get Median Forecast Data for GEOGLoWS IDs\n\nThis tool will get forecast data for a list of GEOGLoWS IDs. The IDs must be a file with only one column. A header may or may not be present. This function will get the data from geoglows V2, select all enesmble medians, and get the maximum values for each ID for all enseumble time. The output file is guarentted to have two columns, one named 'LINKNO' and the other 'max_median_flow'.")
                        id_list_medians  = gr.Textbox(value=manager.default("id_forecast_list"), 
                                                            placeholder='/User/Desktop/id_list.csv',
                                                            label="ID File",
                                                            info=manager.doc('id_forecast_list'),
                                                            )
                        
                        date_median = gr.Textbox(datetime.datetime.now().strftime("%Y%m%d"),
                                            label='Date',
                                            info='Date for forecast data. Must be entered in YYYYMMDD format',
                                            )
                        
                        output_median_file = gr.Textbox(placeholder='/User/Desktop/forecast.csv',
                                                        label='Output File',
                                                        info='Output file for forecast data',
                                                        )
                        
                        get_median_button = gr.Button("Get Max Median Forecast Data")
                        get_median_button.click(fn=manager.get_median_max_forecast, inputs=[id_list_medians, output_median_file, date_median], outputs=[])
                        





    demo.queue().launch(
                server_name="0.0.0.0",
                inbrowser=True,
                quiet=False,
                debug=True,
                show_error=True
            )
