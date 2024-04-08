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
manager = hp.ManagerFacade()

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

with gr.Blocks(title='AutoRoute WebUI') as demo:
    gr.Markdown('# AutoRoute WebUI')
    with gr.Tabs():
        with gr.TabItem('Run AutoRoute'):
            gr.Markdown('## Inputs - Required')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dem = gr.Textbox(placeholder='/User/Desktop/dem.tif',
                                        label="Digital Elevation Model (DEM)",
                                        #info=hp.docs['DEM'],
                                        )
                        dem_name = gr.Textbox(placeholder='Copernicus',
                                            label='DEM Name',
                                            #info=hp.docs['DEM_NAME'],
                                            )
                    with gr.Row():
                        strm_lines = gr.Textbox(placeholder='/User/Desktop/dem.tif',
                                        label="Stream Lines",
                                        #info=hp.docs['DEM'],
                                        )
                        strm_name = gr.Textbox(placeholder='Copernicus',
                                            label='Streamlines Name',
                                            #info=hp.docs['DEM_NAME'],
                                            )
                        flow_id = gr.Dropdown(label='Flow ID',
                                            #info='Specifies the stream identifier that AutoRoute uses. Leave blank to use the first column.',
                                            allow_custom_value=True,
                                            multiselect=False,interactive=True)
                    
                    with gr.Row():
                        lu_file = gr.Textbox(placeholder='/User/Desktop/lu.tif',
                                        label="Land Raster(s)",
                                        #info=hp.docs['DEM'],
                                        )
                        lu_name = gr.Textbox(placeholder='Copernicus',
                                            label='Land Raster Name',
                                            #info=hp.docs['DEM_NAME'],
                                            )
                        mannings_table = gr.Textbox(
                                    placeholder='/User/Desktop/mannings_n.txt',
                                    label="Manning's n table",
                                    #info=hp.docs['LU_Manning_n']
                        )
                
                    with gr.Column():
                        base_max_file = gr.Textbox(
                                    placeholder='/User/Desktop/flow_file.txt',
                                    label="Base and Max Flow File",
                                    #info=hp.docs['Flow_RAPIDFile'],
                        )
                        id_flow_file = gr.Textbox(
                                        placeholder='/User/Desktop/100_year_flow.txt',
                                        label="ID Flow File",
                                        #info=hp.docs['Comid_Flow_File']
                            )
                        with gr.Row():
                            flow_params = gr.Dropdown(label='Flow Columns',
                                                    #info='Specifies the flow rates that AutoRoute uses. Leave blank to use all columns besides the first one.',
                                                    allow_custom_value=True,
                                                    multiselect=True,
                                                    interactive=True)
                            flow_baseflow = gr.Dropdown(label='Base Flow Column',
                                                    #info='Specifies the base flow rates that AutoRoute uses. Leave blank to not use.',
                                                    allow_custom_value=True,
                                                    multiselect=False,
                                                    interactive=True)
                            subtract_baseflow = gr.Checkbox(False,
                                                            label='Subtract Base Flow?',
                                                            interactive=True
                            )
                    
                with gr.Column():
                    data_dir = gr.Textbox(label='Data Directory',
                                        info='Directory where AutoRoute will store its data',
                                        interactive=True)
                    
                    
                    with gr.Row():
                        gr.Markdown("Specify an extent if needed")
                        minx = gr.Number(label='Min X', )
                        maxx = gr.Number(label='Max X', )
                        miny = gr.Number(label='Min Y', )
                        maxy = gr.Number(label='Max Y', )

                    overwrite = gr.Checkbox(label='Overwrite',
                                            #info='Overwrite existing files?',
                                            interactive=True)
                    buffer = gr.Checkbox(label='Buffer',
                                        info='Buffer the DEMs?',
                                        interactive=True)
                    crop = gr.Checkbox(label='Crop',
                                        info='Crop output to extent?',
                                        interactive=True)
                    
                    vdt_file = gr.Textbox(
                                placeholder='/User/Desktop/vdt.txt',
                                label="VDT File",
                                info=hp.docs['vdt']
                    )
                    run_button = gr.Button("Run Model", variant='primary')

                with gr.Row():
                    with gr.Column():
                        depth_map = gr.Textbox(
                            placeholder='/User/Desktop/depth.tif',
                            label="Output Depth Map",
                            #info=hp.docs['out_depth']
                        )
                        flood_map = gr.Textbox(
                            placeholder='/User/Desktop/flood.tif',
                            label="Output Flood Map",
                            #info=hp.docs['out_flood']
                        )
                        velocity_map = gr.Textbox(
                            placeholder='/User/Desktop/velocity.tif',
                            label="Output Velocity Map",
                            #info=hp.docs['out_velocity']
                        )
                        wse_map = gr.Textbox(
                            placeholder='/User/Desktop/wse.tif',
                            label="Output WSE Map",
                            #info=hp.docs['out_wse']
                        )
                        map_button = gr.Button("Preview Extent on Map")

                        map_output = gr.Plot(label="Extent Preview")
                        map_button.click(fn=manager.make_map, inputs=[minx, miny, maxx, maxy], outputs=[map_output])
                    
                
                    
            gr.Markdown('## Inputs - Optional')  
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("AutoRoute parameters", open=False):
                        adjust_flow = gr.Number(1,
                                        label='Adjust Flow',
                                        info=hp.docs['ADJUST_FLOW_BY_FRACTION'],
                                        interactive=True)
                    
                        num_iterations = gr.Number(1,
                                               minimum=1,
                                               label='VDT Database Iterations',
                                               info=hp.docs['num_iterations'],
                                               interactive=True,)
            
                        meta_file = gr.Textbox(
                                        placeholder='/User/Desktop/meta.txt',
                                        label="Meta File",
                                        info=hp.docs['Meta_File']
                            )

                        with gr.Row():
                            convert_cfs_to_cms = gr.Checkbox(False, 
                                                            label='CFS to CMS',
                                                            info='Convert flow values from cubic feet per second to cubic meters per second'
                            )

                        with gr.Row():
                            x_distance = gr.Slider(0,
                                                50_000,
                                                500,
                                                step=1,
                                                label='Cross Section Distance',
                                                info=hp.docs['x_distance'],
                                                interactive=True
                                                )
                            q_limit = gr.Slider(0,
                                                2,
                                                1.1, 
                                                label='Flow Limit',
                                                info=hp.docs['q_limit'],
                                                interactive=True)
                        

                        with gr.Row():
                            direction_distance = gr.Slider(1,500,1,
                                                        step=1,
                                                        label='Direction Distance',
                                                        info=hp.docs['Gen_Dir_Dist'],
                                                        interactive=True)
                            
                            slope_distance = gr.Slider(1,
                                                    500,
                                                    1,
                                                    step=1,
                                                    label='Slope Distance',
                                                        info=hp.docs['Gen_Slope_Dist'],
                                                        interactive=True)
                            
                        with gr.Row():
                            low_spot_distance = gr.Slider(0,500,2,
                                                step=1,
                                                label='Low Spot Distance',
                                                info=hp.docs['Low_Spot_Range'],
                                                interactive=True)
                            with gr.Column():
                                low_spot_is_meters = gr.Checkbox(False, label='Is Meters?')
                                low_spot_use_box = gr.Checkbox(False, label='Use a Range Box?')
                                box_size = gr.Slider(1,10,1,
                                                    step=1,
                                                    label='Box Size',
                                                    visible=False,
                                                    interactive=True)
                                low_spot_use_box.change(lambda x: gr.Slider(visible=x), inputs=low_spot_use_box, outputs=box_size)

                                find_flat = gr.Checkbox(False, label='Find Flat?')
                                low_spot_find_flat_cutoff = gr.Number(float('inf'),
                                                                    label='Flow Cutoff',
                                                                    info='Low_Spot_Find_Flat',
                                                                    visible=False,
                                                                    interactive=True
                                                                    )
                                find_flat.change(lambda x: gr.Number(visible=x), inputs=find_flat, outputs=low_spot_find_flat_cutoff)

                        with gr.Accordion('Sample Additional Cross-Sections', open=False):
                            gr.Markdown(hp.docs['degree'])
                            with gr.Row():
                                degree_manip = gr.Number(0.0, label='Farthest Angle Out (Degree_Manip)')
                                degree_interval = gr.Number(0.0, label='Angle Between Cross-Sections (Degree_Interval)')
                                
                        with gr.Accordion('Set Bounds on Stream Raster', open=False):
                            gr.Markdown(hp.docs['limit_vals'])
                            with gr.Row():
                                Str_Limit_Val = gr.Number(0.0, label='Lowest Perissible Value')
                                UP_Str_Limit_Val = gr.Number(float('inf'), label='Highest Perissible Value')
                        
                        with gr.Row():
                            row_start=gr.Number(0,
                                                precision=0,
                                                label='Starting Row',
                                                info=hp.docs['Layer_Row_Start'])
                            row_end=gr.Number(precision=0,
                                                label='End Row',
                                                info=hp.docs['Layer_Row_End'])
                                
                        with gr.Row():     
                            use_prev_d_4_xs = gr.Dropdown(
                                [0,1],
                                value=1,
                                label='Use Previous Depth for Cross Section',
                                info=hp.docs['use_prev_d_4_xs'],
                                interactive=True
                            )

                            weight_angles = gr.Number(0,
                                            label='Weight Angles',
                                            info=hp.docs['Weight_Angles'],
                                            interactive=True,
                                            )

                            man_n = gr.Number(0.4,
                                            label='Manning\'s n Value',
                                            info=hp.docs['man_n'],
                                            interactive=True,
                                            )
                            
                        lu_name.change(manager.show_mans_n, [lu_name,mannings_table], man_n)

                        with gr.Accordion('Bathymetry', open=False):
                            with gr.Row():
                                with gr.Column():
                                    bathy_file = gr.Textbox(
                                                placeholder='/User/Desktop/bathy.tif',
                                                label="Output Bathymetry File",
                                                info=hp.docs['BATHY_Out_File']
                                    )
                                    bathy_alpha = gr.Number(0.001,
                                                            label='Bathymetry Alpha',
                                                            info=hp.docs['Bathymetry_Alpha'],
                                                            interactive=True,
                                                            )
                                    da_flow_param = gr.Dropdown(label='Drainage or Flow Parameter',
                                                            info=hp.docs['RAPID_DA_or_Flow_Param'],
                                                            allow_custom_value=True,
                                                            multiselect=False,interactive=True)

                                with gr.Column():
                                    bathy_method = gr.Dropdown(['Parabolic', 'Left Bank Quadratic', 'Right Bank Quadratic', 'Double Quadratic', 'Trapezoidal','Triangle'],
                                                            value='Parabolic',
                                                            label='Bathymetry Method',
                                                            info=hp.docs['bathy_method'],
                                                            multiselect=False,interactive=True)
                                    bathy_x_max_depth = gr.Slider(0,1,0.2,
                                                                label='X Max Depth',
                                                                info=hp.docs['bathy_x_max_depth'], 
                                                                visible=False)
                                    bathy_y_shallow = gr.Slider(0,1,0.2,
                                                                label='Y Shallow',
                                                                info=hp.docs['bathy_y_shallow'], 
                                                                visible=False)
                                    
                                    bathy_method.change(manager.bathy_changes, bathy_method, [bathy_x_max_depth, bathy_y_shallow])
                            
                            base_max_file.change(manager.update_flow_params, base_max_file, [flow_id,flow_params, flow_baseflow, da_flow_param])

                with gr.Column():
                    with gr.Accordion("FloodSpreader Parameters", open=False):
                        with gr.Column():
                            omit_outliers = gr.Radio(['None','Flood Bad Cells', 'Use AutoRoute Depths', 'Smooth Water Surface Elevation','Use AutoRoute Depths (StDev)','Specify Depth'],
                                    value='None',
                                    label='Omit Outliers',
                                    interactive=True,
                                    info='None: No outliers will be removed'
                                    )
                            
                            wse_col = gr.Column(visible=False)
                            with wse_col:
                                with gr.Row():
                                    wse_search_dist = gr.Slider(1,100,10,
                                            step=1,
                                            label='Smooth WSE Search Distance',
                                            info=hp.docs['wse_search_dist'],
                                            interactive=True)
                                    wse_threshold = gr.Number(0.25,
                                                              label='Smooth WSE Threshold',
                                                              info=hp.docs['wse_threshold'],
                                                              interactive=True)
                                    wse_remove_three = gr.Checkbox(False,
                                                                   label='Smooth WSE Remove Highest Three',
                                                                   info=hp.docs['wse_remove_three'],
                                                                   interactive=True)
                                    
                            specify_depth = gr.Number(label='Specify Depth',
                                                        interactive=True,
                                                        visible=False)
                            omit_outliers.change(manager.omit_outliers_change, inputs=omit_outliers, outputs=[omit_outliers, wse_col, specify_depth])

                            with gr.Row():
                                twd_factor = gr.Slider(0,10,1.5,
                                                    label='Top Width Distance Factor',
                                                    info=hp.docs['twd_factor'],
                                                    interactive=True)
                                with gr.Column():
                                    only_streams = gr.Checkbox(False,
                                                            label='Only Output Values for Stream Locations',
                                                            info=hp.docs['only_streams'],
                                                            interactive=True)
                                    use_ar_top_widths = gr.Checkbox(False,
                                                            label='Use AutoRoute Top Widths',
                                                            info=hp.docs['use_ar_top_widths'],
                                                            interactive=True)
                                    flood_local = gr.Checkbox(False,
                                                            label='Flood Local',
                                                            info=hp.docs['FloodLocalOnly'],
                                                            interactive=True)
                        

                        with gr.Accordion('Bathymetry', open=False):
                            gr.Markdown('Note that the bathymetry file generated by AutoRoute must be specified in the AutoRoute Bathymetry section')
                            with gr.Row():
                                fs_bathy_file = gr.Textbox(
                                            placeholder='/User/Desktop/floodspreader_bathy.tif',
                                            label="FloodSpreader Output Bathymetry",
                                            info=hp.docs['fs_bathy_file']
                                )
                                with gr.Column():
                                    fs_bathy_smooth_method = gr.Dropdown(['None','Linear Interpolation', 'Inverse-Distance Weighted'],
                                                                         value='None',
                                                                        label='Bathymetry Smoothing',
                                                                        interactive=True) 
                                    bathy_twd_factor = gr.Number(1.0,
                                                                 label='Bathymetry Top Width Distance Factor',
                                                                 interactive=True,
                                                                 visible=False)
                                    fs_bathy_smooth_method.change(lambda x: gr.Number(visible=True) if x[0] == 'I' else gr.Number(visible=False),
                                                                  fs_bathy_smooth_method, bathy_twd_factor)
       
            run_button.click(fn=manager._run,
                                             inputs=[dem,dem_name, strm_lines, strm_name, lu_file, lu_name, base_max_file, subtract_baseflow, flow_id, flow_params, flow_baseflow, num_iterations,
                                                    meta_file, convert_cfs_to_cms, x_distance, q_limit, mannings_table, direction_distance, slope_distance, low_spot_distance, low_spot_is_meters,
                                                    low_spot_use_box, box_size, find_flat, low_spot_find_flat_cutoff, degree_manip, degree_interval, Str_Limit_Val, UP_Str_Limit_Val, row_start, row_end, use_prev_d_4_xs,
                                                    weight_angles, man_n, adjust_flow, bathy_alpha, bathy_file, id_flow_file, omit_outliers, wse_search_dist, wse_threshold, wse_remove_three,
                                                    specify_depth, twd_factor, only_streams, use_ar_top_widths, flood_local, depth_map, flood_map, velocity_map, wse_map, fs_bathy_file, da_flow_param,
                                                    bathy_method,bathy_x_max_depth, bathy_y_shallow, fs_bathy_smooth_method, bathy_twd_factor,
                                                    data_dir, minx, miny, maxx, maxy, overwrite, buffer, crop, vdt_file],
                                                     outputs=[]
                                             )

        with gr.TabItem('File Preprocessing'):
            pass


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
    demo.queue().launch(
                server_name="0.0.0.0",
                inbrowser=True,
                quiet=False,
                debug=True,
                show_error=True
            )
