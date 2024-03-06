import numpy as np
import cv2
from os.path import join
import json
import shape_decomposition2 as sd
from skimage import morphology
import numpy as np

from shapely import ops
from shapely import affinity
from shapely.geometry import LineString,Point
import sas_optimization as so
def myshow(image,name="frame"):
    
    cv2.namedWindow(name,0)
    cv2.imshow(name,image)

    
def medial_axis(img):
    # binaryzation
    _,binary = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    
    # Bone extraction
    binary[binary==255] = 1
    skel, distance =morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8)*255
    
    
    distance= distance.astype(np.uint8)
    
    return dist_on_skel,distance
    
def contains(polygon, cut):
    return polygon.contains(LineString(cut)) or polygon.intersects(LineString(cut))


def cal_data(cut_vector,polygon,skel,distance,drawImg=None):
    """

    Args:
        cut_vector (_type_): 裁剪线段
        polygon (_type_): 轮廓多边形
        skel (_type_): 骨骼标记
        distance (_type_): 骨骼的距离
    """
    
    if not contains(polygon,cut_vector):
        # The clipped line segment is not included in the polygon
        
        return 
    image_h=skel.shape[0]
    # Line clipping
    xx=LineString([Point(p) for p in so.extend_line_segment(cut_vector, 5)])
    children = list(ops.split(polygon, xx))
    if len(children)<2:
        print("useless")
        return -1,0,0
        
    # Order the area from smallest to largest
    children.sort(key=lambda x:-x.area)
    # print("children=",len(children))

    smallChildren=children[1]
    # Acquire small area
    pts=smallChildren.exterior.xy
    pts=[[int(x),int(image_h-y)] for(x,y) in zip(pts[0],pts[1])]



    
    
    maskData=np.zeros(skel.shape[:2],np.uint8)
    cv2.drawContours(maskData,np.array([pts]),0,(255,255,255),-1)

    mask=cv2.bitwise_and(maskData,skel) 
    datamask=cv2.mean(distance,mask)
    
    # The average width of the outline
    avgW=datamask[0]*2+1
    
    length=smallChildren.length
    
    # Small area percentage
    areaRate=smallChildren.area/polygon.area*100

    avgLength=cv2.countNonZero(mask)
    
    lentoW=avgLength/avgW
    
    
    flage=-1
    color=(0,0,0)
    
    if avgLength==0:
        # Indicates a small area that is not inside the figure
        color=(255,255,255)
        flage=-1
    elif areaRate<2:
        # Small area filtration
        color=(0,0,200)
        flage=-1
    elif lentoW>10 and areaRate<10:
        # Length-width ratio
        color=(0,255,200)
        flage=-1
    else:
        flage=1
    # cv2.drawContours(drawImg,np.array([pts]),0,(0,255,0),1) 
    if drawImg is not None and flage<0:
       
        cv2.drawContours(drawImg,np.array([pts]),0,color,-1)
        
        
        # cv2.drawContours(drawImg,np.array([pts]),0,(0,0,255),1)
        
        pass
    
    # print(f"avgW={avgW:.3f},avgLength={avgLength},lentoW={lentoW:.3f},areaRate={areaRate:.4f}")
    return flage,int(avgW),avgLength

def myShapeFilter(input_shape,output_dir):
    # Read json data
    with open(join(output_dir, 'final_cut.json')) as f:
        prediction = json.load(f)
    # Read picture
    img = cv2.imread(input_shape,0)
    # Picture height
    image_h=img.shape[0]

    # Get contour polygon
    polygon = sd.generate_canvas_polygon(img)[0]
    pts=polygon.exterior.xy
    pts=[[int(x),int(image_h-y)] for(x,y) in zip(pts[0],pts[1])]

    # Image axis data extraction
    skel, distance =medial_axis(img)

    # Part of the middle color shows the data
    distanceColor=cv2.cvtColor(distance,cv2.COLOR_GRAY2BGR)
    imageColor=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)


    # Convert data points to integers
    prediction2=np.asarray(prediction).astype(np.int32).tolist()



    imageColor2=imageColor.copy()
    # cal_data(cut_vector,polygon,skel,distance,drawImg=imageColor2)

    # # cv2.putText(imageColor,str(id),(x0,y0),cv2.FONT_HERSHEY_COMPLEX,1,(125,200,10),1)
    final_cuts=[]
    for id,line in enumerate(prediction2):
        # imageColor2=imageColor.copy()
        # Calculate the current profile state
        flag,avgW,avgLength=cal_data(line,polygon,skel,distance,drawImg=imageColor2)
        
        if flag>0:
            final_cuts.append(prediction[id])
            
        
        # print(id,flag,avgW,avgLength)
        x0,y0=line[0]
        x1,y1=line[1]
        y0=image_h-y0


        y1=image_h-y1
        
        # y0,x0=line[0]
        # y1,x1=line[1]
        
        cv2.line(imageColor2,(x0,y0),(x1,y1),(0,0,255),1)
        if flag<0:
            cv2.putText(imageColor2,str(id),((x0+x1)//2,(y0+y1)//2),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,200,10),1)
        else:
            cv2.putText(imageColor2,str(id),((x0+x1)//2,(y0+y1)//2),cv2.FONT_HERSHEY_COMPLEX,0.5,(200,0,200),1)
        # myshow(imageColor2,"imageColor")
        # cv2.waitKey(0)
    # print(len(final_cuts))
    with open(join(output_dir, 'final_cut_opt.json'), 'w') as f:
        json.dump(final_cuts, f,indent=4)
        print(len(final_cuts))
        
    cv2.imwrite(join(output_dir,'my_opt_1.png'),imageColor2)    
    # myshow(distance,"distance")
    # myshow(imageColor2,"imageColor")
    # cv2.waitKey(0)
        
    pass


if __name__ == '__main__':
    # input_image_collection_folder = sys.argv[1]
    # output_dir = sys.argv[2]
    # scaling_factor = int(sys.argv[3])
    
    input_shape="input_data/layout/fly.jpg"
    input_mask_folder="input_data/image_collections/children_mask"
    input_image_collection_folder="input_data/image_collections/children"
    output_dir="output_dir/fly4"
    scaling_factor=2


    
    
    # render_collage(input_image_collection_folder, output_dir, scaling_factor)
    input_shape="input_data/layout/fly.jpg"
    output_dir="output_dir/fly4"





    myShapeFilter(input_shape,output_dir)