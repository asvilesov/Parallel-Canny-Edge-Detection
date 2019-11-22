from utils import utils
import canny_edge_detector as ced
from timeit import default_timer as timer

imgs = utils.load_data()
#utils.visualize(imgs, 'gray')

 


detector = ced.cannyEdgeDetector(imgs, sigma=2, kernel_size=3, lowthreshold=0.09, highthreshold=0.17, weak_pixel=250)


start = timer()

imgs_final = detector.detect()

end = timer()
print(end-start)

utils.visualize(imgs_final, 'gray')