all:lr
lr:lr.cpp
	g++  -o lr -I/usr/local/include/opencv  lr.cpp -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_ts -lopencv_imgcodecs 
