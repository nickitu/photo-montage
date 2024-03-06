import shape_decomposition2 as sd
import sas_optimization3 as so
import collage_assembly as ca
import sys
import my_opt_shape as myopt

if __name__ == '__main__':
    print("shape")
    # input_shape = sys.argv[1]
    # input_mask_folder = sys.argv[2]
    # input_image_collection_folder = sys.argv[3]
    # output_dir = sys.argv[4]
    # scaling_factor = int(sys.argv[5])
    input_shape="input_data/layout/fly.jpg"
    output_dir="output_dir/fly2"
    
    # input_shape="input_data/layout/car.png"
    # output_dir="output_dir/car2"
    
    input_mask_folder="input_data/image_collections/children_mask"
    input_image_collection_folder="input_data/image_collections/children"
    
    scaling_factor=2



    sd.generate_cuts(input_shape, output_dir)
    
    myopt.myShapeFilter(input_shape,output_dir)
    
    
    so.optimization(input_shape, input_mask_folder,input_image_collection_folder, output_dir)
    ca.render_collage(input_image_collection_folder, output_dir, scaling_factor)