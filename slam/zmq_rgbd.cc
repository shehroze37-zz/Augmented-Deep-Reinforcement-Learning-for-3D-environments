#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <string>

#include <zmq.hpp>

#include <opencv2/core/core.hpp>

#include "../../include/System.h"
#include "json.hpp"

#include <sstream>
#include <iostream>

#include <boost/property_tree/json_parser.hpp>
#include <math.h>       /* ceil */
#include <vector>
#include <omp.h>
#include <chrono>

#define PI 3.14159265


using json = nlohmann::json;
using namespace std;

void empty_free(void *data, void *hint) {return;}


class HeatmapsController {
    int width, height;

    //float minX=-256, minY = -256, maxX=1280, maxY=1280 ;
    //float minX=-5672, minY = -12831, maxX=31221, maxY=23781 ;
    //float minX=-5672, minY = -5672, maxX=31221, maxY=23781 ;
    float minX=-12831, minY = -12831, maxX=31221, maxY=31221 ;
    int minXNew, minYNew, maxXNew, maxYNew ;

    
   
    

  public:
 float scaleX, scaleY, padX, padY;
	uint8_t *heatMapsBuffer = NULL;
	    unsigned int heatMapsWidth, heatMapsHeight, heatMapsChannels, heatMapsSize=0, mapSize ;
    HeatmapsController(int, int , int ) ;
    void resetHeatmaps(void) ;
    void updatePlayerHeatMap(cv::Mat, cv::Mat) ;
    void getHeatmap( const string &strVocFile, cv::Mat *heatmap) ;
    void updateWalls(const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &z) ;
    void updateMapObjects(int midpoint_x, int midpoint_y, float depth_value, int map_no, cv::Mat K, cv::Mat inverseCameraPose) ;
};




void HeatmapsController::getHeatmap(const string &heatmap_type, cv::Mat *heatmap){

	if( heatmap_type.compare("walls_player") == 0){
		#pragma omp parallel for
		for(int i = 0; i < heatMapsWidth; i++){
			for(int j = 0; j < heatMapsHeight; j++){
				heatmap->at<float>(i,j) = this->heatMapsBuffer[int(j)*heatMapsWidth + i]  ;				
				//heatmap->at<float>(i,j) = 0.8 * this->heatMapsBuffer[mapSize + (j)*heatMapsWidth + i]  ;
			}	
		}
	}
	else if(heatmap_type.compare("everything") == 0){
		#pragma omp parallel for
		for(int i = 0; i < heatMapsWidth; i++){
			for(int j = 0; j < heatMapsHeight; j++){
				//heatmap->at<float>(i,j) = this->heatMapsBuffer[int(j)*heatMapsWidth + i]  ;				
				heatmap->at<float>(i,j) = 0.8 * this->heatMapsBuffer[mapSize + (j)*heatMapsWidth + i]  ;
				heatmap->at<float>(i,j) = 0.6 * this->heatMapsBuffer[2*mapSize + (j)*heatMapsWidth + i]  ;
				heatmap->at<float>(i,j) = 0.4 * this->heatMapsBuffer[3*mapSize + (j)*heatMapsWidth + i]  ;
				heatmap->at<float>(i,j) = 0.2 * this->heatMapsBuffer[4*mapSize + (j)*heatMapsWidth + i]  ;
			}	
		}
	}


}


void HeatmapsController::updateMapObjects(int midpoint_x, int midpoint_y, float depth_value, int map_no,cv::Mat K, cv::Mat inverseCameraPose){


	cv::Mat u(3,1, CV_32F, double(0)) ;
    	u.at<float>(0,0) = midpoint_x ;
    	u.at<float>(1,0) = midpoint_y ;
    	u.at<float>(2,0) = 1 ;
    						
    	const cv::Mat result = depth_value * (K*u) ;

	cv::Mat V(4,1, CV_32F, double(0)) ;
	V.at<float>(0,0) = result.at<float>(0,0) ;
	V.at<float>(1,0) = result.at<float>(1,0) ;
	V.at<float>(2,0) = result.at<float>(2,0);
    	V.at<float>(3,0) = 1.0 ;

	const cv::Mat final_world_coordinates = inverseCameraPose * V ;
			
	const auto &x = final_world_coordinates.at<float>(0,0);
	const auto &z = final_world_coordinates.at<float>(2,0);
			
	const int x_coordinate = static_cast<int>(x * this->scaleX + this->padX) ;
	const int z_coordinate = static_cast<int>(z * this->scaleY + this->padY) ;

	int posX = x_coordinate * this->scaleX + this->padX;
	int posY = z_coordinate * this->scaleY + this->padY;
	for (int x=-1; x < 2; ++x) {
		        for (int y=-1; y < 2; ++y) {
			    this->heatMapsBuffer[ map_no *mapSize + (posY+y)*heatMapsWidth + posX+x] = 255;
		        }
	}
}

HeatmapsController::HeatmapsController(int width, int height, int channels){

	heatMapsWidth = width ;
	heatMapsHeight = height ;
	heatMapsChannels = channels; 
	int heatMapsSize = heatMapsWidth*heatMapsHeight*heatMapsChannels;
	mapSize = width * height ;

	heatMapsBuffer = (uint8_t*) malloc(heatMapsSize*sizeof(float));
	heatMapsSize = heatMapsWidth * heatMapsHeight * heatMapsChannels ;
        memset(this->heatMapsBuffer, 0, heatMapsSize);


	scaleX = - float(heatMapsWidth-4) / float(maxX - minX);
        scaleY = float(heatMapsHeight-4) / float(maxY - minY);
        padX = -2 - minX * scaleX + heatMapsWidth;
        padY = 2 - minY * scaleY;

	this->maxXNew = 0 ;
	this->maxYNew = 0 ;
	this->minXNew = 0 ;
	this->minYNew = 0 ;

}

void HeatmapsController::updatePlayerHeatMap(cv::Mat pose, cv::Mat K){

	float alpha = atan2(pose.at<float>(1,0) , pose.at<float>(0,0)) ;
	float beta = atan2((-1 * pose.at<float>(2,0)) , sqrt((pose.at<float>(2,1) * pose.at<float>(2,1)) + (pose.at<float>(2,2) * pose.at<float>(2,2)))) ; ;
	float gemma = atan2(pose.at<float>(1,0) , pose.at<float>(0,0)) ;

	alpha = alpha * 180 / PI ;
	beta = beta * 180 / PI ;
	gemma = gemma * 180 / PI ;
	float playerAngle = beta * 1;
	if (playerAngle < 0){
		playerAngle = playerAngle * -1 ;
	}

	//printf("PlayerAngle = %4.2f\n", playerAngle) ;
	//printf("Alpha = %4.2f, Gamma = %4.2f and beta = %4.2f\n", alpha,gemma,beta) ;	
	int playerX = (int) pose.at<float>(0,3), playerY = (int) pose.at<float>(1,3), playerZ = (int) pose.at<float>(2,3) ;

	/*printf("Player X = %4.2f, ", pose.at<float>(0,3)) ;
	printf("player Y = %4.2f ",  pose.at<float>(1,3)) ;
	printf("Player Z = %4.2f\n", pose.at<float>(2,3)) ;

	printf("Player X = %i, ", playerX) ;
	printf("player Y = %i ",  playerY) ;
	printf("Player Z = %i\n", playerZ) ;*/

	playerY = playerZ ;

	if (playerY < 0 && playerY < this->minYNew){
		this->minYNew = playerY ;
	}

	if (playerY > 0 && playerY > this->maxYNew){
		this->maxYNew = playerY ;
	}

	if (playerX < 0 && playerX < this->minXNew){
		this->minXNew = playerX ;
	}


	if (playerX > 0 && playerX > this->maxXNew){
		this->maxXNew = playerX ;
	}

        playerX = playerX * this->scaleX + this->padX;
        playerY = playerY * this->scaleY + this->padY;

                
	memset(this->heatMapsBuffer+mapSize, 0, mapSize);

            int centerValue = 255;
            int arrowValue = 125;
            this->heatMapsBuffer[mapSize + playerY*heatMapsWidth + playerX] = centerValue;
            playerAngle = int(360 - playerAngle + 180) % 360;
            if (playerAngle < 22.5) {
                this->heatMapsBuffer[mapSize + (playerY+0)*heatMapsWidth + playerX+1] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY+0)*heatMapsWidth + playerX+2] = arrowValue;

            } else if (playerAngle < 67.5) {
                this->heatMapsBuffer[mapSize + (playerY+1)*heatMapsWidth + playerX+1] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY+2)*heatMapsWidth + playerX+2] = arrowValue;
            } else if (playerAngle < 112.5) {
                this->heatMapsBuffer[mapSize + (playerY+1)*heatMapsWidth + playerX+0] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY+2)*heatMapsWidth + playerX+0] = arrowValue;

            } else if (playerAngle < 157.5) {
                this->heatMapsBuffer[mapSize + (playerY+1)*heatMapsWidth + playerX-1] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY+2)*heatMapsWidth + playerX-2] = arrowValue;

            } else if (playerAngle < 202.5) {
                this->heatMapsBuffer[mapSize + (playerY+0)*heatMapsWidth + playerX-1] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY+0)*heatMapsWidth + playerX-2] = arrowValue;
            } else if (playerAngle < 247.5) {
                this->heatMapsBuffer[mapSize + (playerY-1)*heatMapsWidth + playerX-1] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY-2)*heatMapsWidth + playerX-2] = arrowValue;

            } else if (playerAngle < 292.5) {
                this->heatMapsBuffer[mapSize + (playerY-1)*heatMapsWidth + playerX+0] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY-2)*heatMapsWidth + playerX+0] = arrowValue;		

            } else if (playerAngle < 337.5) {
                this->heatMapsBuffer[mapSize + (playerY-1)*heatMapsWidth + playerX+1] = arrowValue;
                this->heatMapsBuffer[mapSize + (playerY-2)*heatMapsWidth + playerX+2] = arrowValue;

            }

	   /*printf("Player X Max = %i, ",  this->maxXNew) ;
	   printf("Player X Min = %i, ",  this->minXNew) ;
	   printf("PLayer Y max = %i, ",  this->maxYNew) ;
	   printf("player Y Min = %i \n", this->minYNew) ;*/
        
}

void HeatmapsController::updateWalls(const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &z){

	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++){

		const auto &height = y[i];
		if (height > -1100 && height < -1000)
		{
			const int x_coordinate = static_cast<int>(x[i] * this->scaleX + this->padX) ;
			const int z_coordinate = static_cast<int>(z[i] * this->scaleY + this->padY) ;

			#pragma omp critical
			{
				this->heatMapsBuffer[z_coordinate * heatMapsWidth + x_coordinate] = 255;
			}
		}
	}
}	

void HeatmapsController::resetHeatmaps(){	
	memset(this->heatMapsBuffer, 0, heatMapsSize);
	printf("Reseting heatmaps\n") ;
}


int main(int argc, char **argv)
{

    //omp_set_num_threads(12) ;
    int count = 1 ;
    if(argc != 5)
    {
        cerr << endl << "Usage: ./zmq_rgbd port path_to_vocabulary path_to_settings" << endl;
        return 1;
    }
    
    HeatmapsController heatmaps_controller(84,84,19) ;

    cv::Mat K(3,3, CV_32F, double(0)) ;
    K.at<float>(0,0) = 191.999512;
    K.at<float>(1,1) = 191.999512;
    K.at<float>(2,2) = 1;
    K.at<float>(0,2) = 160;
    K.at<float>(1,2) = 120;
    K = K.inv() ;


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[2],argv[3],ORB_SLAM2::System::RGBD,true);

    printf("Starting\n") ;
    // Initialize zmq
    zmq::context_t ctx(1);
    zmq::socket_t socket (ctx, ZMQ_REP);

    std::string addr = "tcp://*:";
    addr.append(argv[1]);
    socket.bind(addr.c_str());
    std::cout << "Bound on " << addr << std::endl;

    cv::Mat rgb_img, d_image;
    for(;;) {

        // zmq recv
        zmq::message_t mr_json;
    	zmq::message_t object_detector ;

        zmq::message_t mr_data_rgb;
        zmq::message_t mr_data_d;
        socket.recv(&mr_json);
        std::string temp((const char*)mr_json.data(), mr_json.size());
        json metadata = json::parse(temp);

        socket.recv(&mr_data_rgb);
        socket.recv(&mr_data_d);

    	socket.recv(&object_detector) ;
    	std::string temp_2((const char*)object_detector.data(), object_detector.size()) ;
    	json object_detector_data = json::parse(temp_2) ;
	
	if (metadata["shape"][0] == 1){
		printf("Resetting\n") ;
		SLAM.Reset() ;
		heatmaps_controller.resetHeatmaps() ;
		cv::Mat heatmap(84,84,CV_64F, double(0)) ;
		json out_metadata = {
		    {"shape", {heatmap.cols, heatmap.rows}},
		    {"dtype", "float32"},
		};
		std::string out_metadata_buff = out_metadata.dump();
		zmq::message_t ms_json((void*)out_metadata_buff.c_str(), out_metadata_buff.size(), empty_free);
		zmq::message_t ms_data((void*)heatmap.data, heatmap.cols*heatmap.rows*4, empty_free); // 4=sizeof(float32)
		socket.send(ms_json, ZMQ_SNDMORE);
		socket.send(ms_data);


		continue ;
	}

        // to cv::Mat
        cv::Mat rgb_img(metadata["shape"][0], metadata["shape"][1], CV_8UC3, mr_data_rgb.data());
        cv::Mat d_img(metadata["shape"][0], metadata["shape"][1], CV_16UC1, mr_data_d.data());

	cv::Mat pose = SLAM.TrackRGBD(rgb_img,d_img,metadata["timer"]);
	cv::Mat inverseCameraPose = pose.inv() ;
	inverseCameraPose.at<float>(0,3) = 5000 * inverseCameraPose.at<float>(0,3) ;
    	inverseCameraPose.at<float>(1,3) = 5000 * inverseCameraPose.at<float>(1,3) ;
    	inverseCameraPose.at<float>(2,3) = 5000 * inverseCameraPose.at<float>(2,3) ;
    	inverseCameraPose.at<float>(3,3) = 5000 * inverseCameraPose.at<float>(3,3) ;

	//remove objects from depth
	const string heatmap_type = argv[4] ;
    	for (json::iterator it = object_detector_data.begin(); it != object_detector_data.end(); ++it) {
      		//std::cout << it.key() << " : " << it.value() << "\n";
    		std::cout << it.value() << "\n";
    		int x1 = (int) object_detector_data[it.key()][1];
    		int y1 = (int) object_detector_data[it.key()][2];
    		int x2 = (int) object_detector_data[it.key()][3];
    		int y2 = (int) object_detector_data[it.key()][4];


		//'health', 'monster', 'high_grade_weapons', 'high_grade_ammo', 'other_ammo', 'my_shoots','monster_shoots'
		if (heatmap_type.compare("everything") == 0){

			int midpoint_x = (x2 - x1) / 2 ;
			int midpoint_y = (y2 - y1) / 2 ;
			float depth_value = d_img.at<float>(midpoint_x, midpoint_y) ;
			const string object_type = object_detector_data[it.key()][0] ;
			int map_no = 0 ;
			if (object_type.compare("health") == 0){
				map_no = 2 ;
			}
			else if(object_type.compare("high_grade_weapons") == 0 || object_type.compare("high_grade_ammo") == 0 || object_type.compare("other_ammo") == 0){
				map_no = 3 ;
			}
			else if(object_type.compare("monster") == 0){
				map_no = 4 ;
			}
			
			if(map_no != 0){
				heatmaps_controller.updateMapObjects(midpoint_x, midpoint_y,depth_value, map_no, K, inverseCameraPose ) ;
			}

		}


		#pragma omp parallel for
		for (int i = x1; i <= x2; i++){
			for(int j = y1; j <= y2 ; j++){
				d_img.at<float>(i,j) = 0 ;
			}
		}

    	}


	//printf("[(%4.2f, %4.2f,%4.2f,%4.2f)(%4.2f,%4.2f,%4.2f,%4.2f)(%4.2f,%4.2f,%4.2f,%4.2f)(%4.2f,%4.2f,%4.2f,%4.2f)]\n",pose.at<float>(0,0),pose.at<float>(0,1),pose.at<float>(0,2),pose.at<float>(0,3),pose.at<float>(1,0),pose.at<float>(1,1),pose.at<float>(1, 2),pose.at<float>(1,3),pose.at<float>(2,0),pose.at<float>(2,1),pose.at<float>(2,2),pose.at<float>(2,3),pose.at<float>(3,0),pose.at<float>(3,1),pose.at<float>(3,2),pose.at<float>(3,3) );

	
    	heatmaps_controller.updatePlayerHeatMap(inverseCameraPose, K) ;
	
    	//calculate new walls
	//printf("calculating walls\n") ;
        std::vector<float> xCoordinates(d_img.rows * d_img.cols, 0.0);
        std::vector<float> yCoordinates(d_img.rows * d_img.cols, 0.0);
        std::vector<float> zCoordinates(d_img.rows * d_img.cols, 0.0);

	d_img.convertTo(d_img, CV_32F);

	/*
	ofstream cameraMatrix ;
	if (count == 1){
		cameraMatrix.open("CameraMatrix.txt") ;
		cameraMatrix << fixed ;  
		cameraMatrix.close() ;
	}
	*/
  	 
    	const auto c_start = std::chrono::system_clock::now();
 
	#pragma omp parallel for
    	for(int row = 0; row < d_img.rows; row++){
    		for(int col = 0; col < d_img.cols; col++){

    			cv::Mat u(3,1, CV_32F, double(0)) ;
    			u.at<float>(0,0) = col ;
    			u.at<float>(1,0) = row ;
    			u.at<float>(2,0) = 1 ;
    						
			const float depth_value = d_img.at<float>(row,col) ;
    			const cv::Mat result = depth_value * (K*u) ;
			//printf("Result calculated\n") ;

			

			cv::Mat V(4,1, CV_32F, double(0)) ;
			V.at<float>(0,0) = result.at<float>(0,0) ;
			V.at<float>(1,0) = result.at<float>(1,0) ;
			V.at<float>(2,0) = result.at<float>(2,0);
    			V.at<float>(3,0) = 1.0 ;

			const cv::Mat final_world_coordinates = inverseCameraPose * V ;

			const auto &height = final_world_coordinates.at<float>(1,0);
			if (height < -1100 && height > -1000)
				continue;			
#if 0

			const auto &IC = inverseCameraPose;
			const auto &r = result ;
			const float x = IC.at<float>(0,0) * r.at<float>(0,0) + IC.at<float>(0,1) * r.at<float>(1,0) + IC.at<float>(0,2) * r.at<float>(2,0) + IC.at<float>(0,3);
			const float z = IC.at<float>(2,0) * r.at<float>(0,0) + IC.at<float>(2,1) * r.at<float>(1,0) + IC.at<float>(2,2) * r.at<float>(2,0) + IC.at<float>(2,3);

			const auto &f = inverseCameraPose;
			if(std::abs(f.at<float>(3, 0) - 0.0) > 1e-10 ||
			   std::abs(f.at<float>(3, 1) - 0.0) > 1e-10 ||
		 	   std::abs(f.at<float>(3, 2) - 0.0) > 1e-10 ||
			   std::abs(f.at<float>(3, 3) - 5000.0) > 1e-10)
				std::cout << "incorrect last row in CMatrix: " << f.row(3) << std::endl;

			if(std::abs(final_world_coordinates.at<float>(3, 0) - 5000.0) > 1e-10)
				std::cout << "prob incorrect result: " << final_world_coordinates.at<float>(3, 0) << std::endl;
#endif

			const auto &x = final_world_coordinates.at<float>(0,0);
			const auto &z = final_world_coordinates.at<float>(2,0);
			
			const int x_coordinate = static_cast<int>(x * heatmaps_controller.scaleX + heatmaps_controller.padX) ;
			const int z_coordinate = static_cast<int>(z * heatmaps_controller.scaleY + heatmaps_controller.padY) ;

			#pragma omp critical
			{
				heatmaps_controller.heatMapsBuffer[z_coordinate * heatmaps_controller.heatMapsWidth + x_coordinate] = 255;
			}

/*
			const auto &height = final_world_coordinates.at<float>(1,0);
			const auto &x = final_world_coordinates.at<float>(0,0);
			const auto &z = final_world_coordinates.at<float>(2,0);
			if (height > -1100 && height < -1000)
			{
				const int x_coordinate = static_cast<int>(x * heatmaps_controller.scaleX + heatmaps_controller.padX) ;
				const int z_coordinate = static_cast<int>(z * heatmaps_controller.scaleY + heatmaps_controller.padY) ;

				#pragma omp critical
				{
					heatmaps_controller.heatMapsBuffer[z_coordinate * heatmaps_controller.heatMapsWidth + x_coordinate] = 255;
				}
			}
*/
		}
    	}

	//heatmaps_controller.updateWalls(xCoordinates, yCoordinates, zCoordinates) ;

	const auto c_end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = c_end - c_start;
	std::cout << "chrono elapsed time: " << elapsed_seconds.count() << "s\n";


        // walls finished
        cv::Mat heatmap(84,84,CV_64F, double(0)) ;
	/*float totalSumBefore = 0 ;
	for (int i = 0; i < heatmap.rows;i++){
		for(int j = 0; j < heatmap.cols; j++){
			totalSumBefore += heatmap.at<float>(i,j) ;	
		}
	}*/
	//printf("Total sum before = %4.2f\n", totalSumBefore) ;
	heatmaps_controller.getHeatmap( argv[4], &heatmap) ;



	/*for (int i = 5; i < 10;i++){
		for(int j = 5; j < 10; j++){
				heatmap.at<float>(i,j) = 1 ;
		}
	}

	totalSumBefore = 0 ;
	for (int i = 0; i < heatmap.rows;i++){
		for(int j = 0; j < heatmap.cols; j++){
			totalSumBefore += heatmap.at<float>(i,j) ;	
		}
	}
	printf("ELement at 6,6 = %4.2f\n", heatmap.at<float>(6,6)) ;
	printf("Total sum after = %4.2f\n", totalSumBefore) ;*/


        // zmq send
        json out_metadata = {
            {"shape", {heatmap.cols, heatmap.rows}},
            {"dtype", "float32"},
        };
        std::string out_metadata_buff = out_metadata.dump();
        zmq::message_t ms_json((void*)out_metadata_buff.c_str(), out_metadata_buff.size(), empty_free);
        zmq::message_t ms_data((void*)heatmap.data, heatmap.cols*heatmap.rows*4, empty_free); // 4=sizeof(float32)
        socket.send(ms_json, ZMQ_SNDMORE);
        socket.send(ms_data);
        printf("Answer sent, looping.\n");

	count++ ;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
