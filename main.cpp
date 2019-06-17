//
//  main.cpp
//
//
//  Created by Benjamin Barral and Elie Oriol on 05/03/2016.
//

/// C++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <math.h>
/// OpenCv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
/// Sound : AQUILA
#include <SFML/Audio.hpp>
#include <SFML/System.hpp>
#include <aquila/global.h>
#include <aquila/transform.h>

using namespace cv;
using namespace std;


/// IMAGE PROCESSING PARAMETERS AND VARIABLES
// Canny edge detection parameters
int low_threshold = 10;
int thresh_ratio = 3;
int kernel_size = 3;
// Point filtering (min distance between Delaunay points, min length for contour recursion)
const int min_pt_offset = 6;
const float min_contour_length = 40.;
// Triangle filtering : filter out large triangles (usually between the person's contour and the background)
const float max_triangle_size = 140.;
// Image parameters : display resolution, processing resolution
const int disp_width = 640, disp_height = 480, proc_width = 640,  proc_height = 480;
// Objects used in the image processing pipeline
Mat im_src, src, src_gray; 
Mat detected_edges;
vector< vector<Point> > connected_contours;
Ptr<BackgroundSubtractor> pMOG2; // Object for OpenCV's "MOG2" Background subtractor
Mat foreground_mask_MOG2; // foreground mask
vector< vector<Point> > background_ptsX, foreground_ptsX, total_ptsX; // Delaunay points sorted in the X direction
vector<Point> convex_hull;
vector<Vec6f> triangles;
Mat rasterized_image; // Result image
// Background : original or rasterized
bool rasterize_background_mode = true;


/// DISTORTION EFFECTS PARAMETERS AND VARIABLES
const int index_beat_effect=1;
const int key_beat_effect = 1; // BEAT
const int key_note_effect = 13; // Music note
bool activate_effect [4] = {false,false,false,false}; // 4 distortion effects
// Time parameters
clock_t clock_start;
int frame_counter = 0;
// Horizontal compression wave [effect 1]
int wave_front1 = -1; // front of the wave
const int wave_width1 = 60;   // width of the wave
const int wave_speed1 = 50;  // speed of the wave
// Circular compression wave [effect 2]
int radial_front = -1;
Point wave2_origin;                   // origin of the wave
const int wave_width2 = 60;
vector<Point> conv_hull2;        //convex hull when the effect was triggered
const int wave_speed2 = 40;
const float wave_compr_rate = 0.8;           // compression ratio of the wave
// Perpendicular compression wave [effect 3]
int wave_front3 = -1;
const int wave3_param1 = 80;                   //width covered by the wave in x direction
const int wave3_param2 = 20;                  //width of the compression
const int wave3_param3 = 100;                 //width transformed by the wave in y direction
const int wave3_param4 = 60;
const int wave_speed3 = 60;
int wave_Y3;          // Y coordinate of the wave
// Circular expansion effect --> effect of widening triangles --> blur [effect4]
int time_wave4 = -1;
Point wave4_origin;
const int wave4_param1 = 55;                   // range of points covered by the effect
int wave4_curr_width;               // effect spreads to R+largeur3
const int wave4_max_width = 120;
const float wave4_freq = 0.11;
vector<Point> conv_hull4;
// Circular compression effect --> effect of crushing the triangles --> fine detail [effect 5]
int wave5_time = -1;
Point wave5_origin;
const int wave5_max_front=125;
int wave5_curr_front;
const float wave5_freq = 0.09;      // speed
vector<Point> conv_hull5;
// Foreground Compression effect [effect 6]
int wave6_time =-1;
Point wave6_origin;
int wave6_range1, wave6_range2;
const float wave6_freq = 0.06;
vector<Point> conv_hull6;        //convex hull at instant of spark of the effect
// Matrix that says if a point in the image is currently being "distorted"
bool is_point_distorted[proc_width][proc_height];

/// MUSIC and RHYTHM DETECTION PARAMETERS AND VARIABLES
// Sound interaction : triggers the effect in reaction to the beats
bool sound_interaction_mode = true; // set to false if you wanna trigger the effects manually instead of automatically with rythm detection
bool play_music = true;        // set to false if you don't want to hear the music
const int delay_play=0;
// Path of the audio file
string path_track = "../The XX   Intro [long version].wav";
// Rythm detection
bool only_beat_mode = false;
bool low_freq_detection_mode = true;
const int n_low_frequencies = 6; // number of frequencies to parse for beat detection
int bpm = 100; // BPM of the song (beats per minute) : average estimation - Write the real BPM if known
double current_sound_intensity;
static int n_intensity_blocks = 43;
double mean_buffer_intensity;
double variance_buffer_intensity;
double time_since_last_beat;
const float c1 = -0.0000015f;
const float c2 = 1.5142857f; // parameters for the beat detection formula : following M. Ziccardi's formula : http://mziccard.me/2015/05/28/beats-detection-algorithms-1/
int buffer_counter;
const double coef_beat = 2.;
const double ratio_beat = 0.6; // parameters for beat detection
const double current_to_mean_rate_beat = 1.5;
const double current_to_mean_note = 0.5; // experimental ratios for rhythm detection filtering


/// FUNCTIONS
// X value comparator
bool compX(const Point& lhs, const Point& rhs) {
    return lhs.x < rhs.x;
}
// Y value comparator
bool compY(const Point& lhs, const Point& rhs) {
    return lhs.y < rhs.y;
}

// Convert to gray, blur and perform Canny edge detection
void DetectEdges() {
    /// Update the background model
    pMOG2->apply(im_src, foreground_mask_MOG2, 0);
    
    src = Scalar::all(0);                   // If trip, superpose sources
    
    im_src.copyTo(src);
    
    /// Grayscale & initialize rasterized_image
    cvtColor( src, src_gray, CV_BGR2GRAY);
    rasterized_image.create( src.size(), src.type() );
    
    /// Reduce noise with two kernels
    blur( src_gray, detected_edges, Size(5,5) );
    blur( detected_edges, detected_edges, Size(3,3) );
    
    /// Canny detector & find contours
    Canny( detected_edges, detected_edges, low_threshold, low_threshold*thresh_ratio, kernel_size );
    findContours(detected_edges, connected_contours, noArray(), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0,0));
}


// Add regular points on the detected contours, recursively
void AddContourPoints(const vector<Point>& contour) {
    int l = contour.size();
    
    /// Add point in the middle if long enough. If not, return.
    if (l >= min_contour_length) {
        bool goodPt = true;
        for (int j = -min_pt_offset; j <= min_pt_offset; j++) {
            vector<Point> ptsj = total_ptsX[min(max(contour[l/2].x + j,0),src.cols-1)];
            if (ptsj.size() != 0) {
                for (Point pt : ptsj) {
                    if (pt.y >= contour[l/2].y - min_pt_offset && pt.y <= contour[l/2].y + min_pt_offset) {
                        goodPt = false;
                        break;
                    }
                }
            }
            if (!goodPt)
                break;
        }
        if (goodPt) {
            total_ptsX[contour[l/2].x].push_back(contour[l/2]);
            if (foreground_mask_MOG2.at<uchar>(contour[l/2].y, contour[l/2].x) == 0)
                background_ptsX[contour[l/2].x].push_back(contour[l/2]);
            else
                foreground_ptsX[contour[l/2].x].push_back(contour[l/2]);
        }
        
    }
    else
        return;
    
    /// Call lengthPts on contour[begin, mid] & contour[mid, end]
    vector<Point>::const_iterator mid = contour.begin() + l/2;
    vector<Point> c1(contour.begin(), mid), c2(mid, contour.end());
    AddContourPoints(c1);
    AddContourPoints(c2);
}

// Find Delaunay points using contours
void DelaunayPts()
{
    /// Initialize ptsX
    total_ptsX.clear();
    total_ptsX.resize(src.cols);
    foreground_ptsX.clear();
    foreground_ptsX.resize(src.cols);
    background_ptsX.clear();
    background_ptsX.resize(src.cols);
    
    random_shuffle(connected_contours.begin(), connected_contours.end());
    
    /// For each contour, take the extreme points (left, right, top, bottom) and add points in the middle if it is a long contour
    for (int k = 0; k < connected_contours.size(); k++) {
        Point extLeft  = *min_element(connected_contours[k].begin(), connected_contours[k].end(), compX);
        Point extRight = *max_element(connected_contours[k].begin(), connected_contours[k].end(), compX);
        Point extTop   = *min_element(connected_contours[k].begin(), connected_contours[k].end(), compY);
        Point extBot   = *max_element(connected_contours[k].begin(), connected_contours[k].end(), compY);
        Point ext[4] = {extLeft, extRight, extTop, extBot};
        
        for (int i = 0; i < 4; i++) {
            bool goodPt = true;
            for (int j = -min_pt_offset; j <=min_pt_offset; j++) {
                vector<Point> ptsj = total_ptsX[min(max(ext[i].x + j,0),src.cols-1)];
                if (ptsj.size() != 0) {
                    for (Point pt : ptsj) {
                        if (pt.y >= ext[i].y - min_pt_offset && pt.y <= ext[i].y + min_pt_offset) {
                            goodPt = false;
                            break;
                        }
                    }
                }
                if (!goodPt)
                    break;
            }
            if (goodPt) {
                total_ptsX[ext[i].x].push_back(ext[i]);
                if (foreground_mask_MOG2.at<uchar>(ext[i].y, ext[i].x) == 0)
                    background_ptsX[ext[i].x].push_back(ext[i]);
                else
                    foreground_ptsX[ext[i].x].push_back(ext[i]);
            }
        }
        
        AddContourPoints(connected_contours[k]);
    }
}

// Use Delaunay triangulation and the filtered points to rasterize the image
void Triangulate() {
    //Initialize hull
    //hull.clear();
    
    /// Define subdiv
    Rect rect(0, 0, rasterized_image.size().width, rasterized_image.size().height);
    Subdiv2D subdiv(rect);
    
    /// Add Delaunay points to subdiv & calculate convex hull
    vector<Point> dPts;
    for (vector<Point> v: foreground_ptsX) {
        for (Point pt: v) {
            dPts.push_back(pt);
            subdiv.insert(pt);
        }
    }
    convexHull(Mat(dPts), convex_hull);
    
    /// Add background points to subdiv & move points from bgptsX that are in the convex hull to dptsX
    for (vector<Point> v: background_ptsX) {
        for (int i = 0; i < v.size(); i++) {
            if (pointPolygonTest(Mat(convex_hull), v[i], false) >= 0) {
                foreground_ptsX[v[i].x].push_back(v[i]);
                v.erase(v.begin() + i);
            }
            subdiv.insert(v[i]);
        }
    }
    
    /// Triangulation
    subdiv.getTriangleList(triangles);
}

// Fill the triangles with the average color
void ColorizeTriangles(float timeFactor = 0.5, int xo = rasterized_image.cols/2, int yo = rasterized_image.rows/2, int r = 60, Scalar sc = Scalar(0, 0, 64)) {
    
    /// fillPoly parameters
    Point triPts[1][3], triPts2[1][3];
    int npts = 3;
    int x[3], y[3], xmin, ymin, xmax, ymax;
    int longTr=0;   //maximal side length of a triangle
    
    /// For each triangle, colorize it with source frame's average color inside it
    for (Vec6f t : triangles) {
        bool colorModulation = false;
        bool out = false;
        
        /// Define x and y values of triangle's points
        /// If one point of the triangle is out the convex hull, out is set to true
        for (int i = 0; i < 3; i++) {
            x[i] = t[2*i];
            y[i] = t[2*i+1];
            
            if (pointPolygonTest(Mat(convex_hull), Point(x[i],y[i]), false) < 0) out = true;
            for (int j=0; j<i; j++){
                if (norm(Point(x[i],y[i])-Point(x[j],y[j]))>longTr) longTr=norm(Point(x[i],y[i])-Point(x[j],y[j]));
            }
        }
        
        if (!rasterize_background_mode && out ==false){
            bool inEffect=false;
            for (int i = 0; i < 3; i++) {
                if (is_point_distorted[x[i]][y[i]]==true){
                    inEffect=true; //if one of the three points is in an effect, we keep the triangle, even if it is too big
                }
            }
            if (!inEffect && longTr>max_triangle_size ) out =true;//we get rid of the triangles that are too big and are not part of the effect = triangles at the limit of the body but which are in the hull
        }
        longTr=0;
        triPts2[0][0] = Point(t[0], t[1]);
        triPts2[0][1] = Point(t[2], t[3]);
        triPts2[0][2] = Point(t[4], t[5]);
        const Point* triangle2[1] = {triPts2[0]};
        
        if (rasterize_background_mode || !out) {
            /// Define region of interest in source frame
            xmin = *min_element(x,x+3);
            if (xmin < 0)
                xmin = 0;
            
            ymin = *min_element(y,y+3);
            if (ymin < 0)
                ymin = 0;
            
            xmax = *max_element(x,x+3);
            if (xmax >= src.cols)
                xmax = src.cols - 1;
            
            ymax = *max_element(y,y+3);
            if (ymax >= src.rows)
                ymax = src.rows - 1;
            
            triPts[0][0] = Point(t[0], t[1]) - Point(xmin, ymin);
            triPts[0][1] = Point(t[2], t[3]) - Point(xmin, ymin);
            triPts[0][2] = Point(t[4], t[5]) - Point(xmin, ymin);
            const Point* triangle[1] = {triPts[0]};
            
            Mat roi = im_src(Range(ymin, ymax+1), Range(xmin,xmax+1));
            
            /// Create mask and calculate average color in region of interest with mask
            Mat1b mask(roi.rows, roi.cols, uchar(0));
            
            fillPoly(mask, triangle, &npts, 1, Scalar(255));
            
            Scalar avg = mean(roi, mask), avgMod;
            
            /// If colorModulation is true, modulate color of the triangle
            if (colorModulation) {
                for (int k = 0; k < 3; k++) {
                    avgMod[k] = timeFactor*avg[k] + (1-timeFactor)*sc.val[k];
                }
            }
            
            /// fillPoly with color (modulated or not) in dst
            if (colorModulation)
                fillPoly(rasterized_image, triangle2, &npts, 1, avgMod);
            else
                fillPoly(rasterized_image, triangle2, &npts, 1, avg);
        }
        
        else
            fillPoly(rasterized_image, triangle2, &npts, 1, Scalar(0));
    }
}

// Distortion effects functions
//Horizontal compression wave
int CompressHorizontallyPoint(const int& x, int a, float compr, int largeur){//moves the points in [a,a+largeur] to [a+compr*largeur,a+largeur]
    int c=a+int(compr*largeur);
    int b=a+largeur;
    return int((b*(c-a)+x*(b-c))/(b-a));
}

vector< vector<Point> > CompressHorizontallyImage(const vector<vector<Point> >& pts, int xBegin, float compr, int largeur){
    if (xBegin>=proc_width-largeur) largeur=proc_width-xBegin-1;
    vector< vector<Point> > ptsCp(pts);
    for (int i=xBegin;  i<xBegin+largeur; i++){
        ptsCp[i]=vector<Point>();
    }
    for (int i=xBegin;  i<xBegin+largeur; i++){
        vector<Point> ptsI=total_ptsX[i];
        for (Point p:ptsI){
            ptsCp[CompressHorizontallyPoint(i,xBegin,compr, largeur)].push_back(Point(CompressHorizontallyPoint(i, xBegin, compr,largeur),p.y)); //we modify the points of the area of the wave
        }
    }
    return ptsCp;
}


//Perpendicular compression wave
int fOrdonnéeVaguePerpendiculaire(int x,int xStart,int y,int yStart,int l1, int l2, int L){//moves the points in [yStart-l2,yStart+l2] to [yStart-l1,yStart+l1]
    int res;
    res=yStart+(y-yStart)*float(l1)/l2;
    if (res>=proc_height) res=proc_height-1;
    if (res<0) res=0;
    return res;
}

vector< vector<Point> > translationPerpendiculaire(const vector<vector<Point> >& pts, int xStart, int yStart,int l1, int l2, int L){
    if (xStart>=proc_width-L) L=proc_width-xStart-1;
    if (yStart>=proc_height-l2) l2=proc_height-yStart-1;
    if (yStart<l2) l2=yStart-1;
    vector< vector<Point> > ptsCp(pts);
    for (int i=xStart;  i<xStart+L+1; i++){
        ptsCp[i]=vector<Point>();
    }
    Point q;
    for (int i=xStart;  i<xStart+L+1; i++){
        vector<Point> ptsI=total_ptsX[i];
        for (Point p:ptsI){
            if (p.y>=yStart-l2 && p.y<=yStart+l2){
                q=Point(p.x,fOrdonnéeVaguePerpendiculaire(i, xStart, p.y, yStart, l1, l2, L));
                ptsCp[i].push_back(q);//we modify the points of the area of the wave
                is_point_distorted[q.x][q.y]=true;
            }
            else  ptsCp[i].push_back(p);
        }
    }
    return ptsCp;
}


//Circular compression wave
Point fComprRadiale(const Point& z, const Point& o, int a, float compr, int largeur){//moves the points from radius [a,a+largeur] to [a+compr*largeur,a+largeur]
    Point zbis=z-o;
    float r=norm(zbis);
    float rtild=CompressHorizontallyPoint(r, a, compr, largeur);
    return o+Point(int(rtild*zbis.x/r),int(rtild*zbis.y/r));
}

int distBord(const Point& o, const Point& p){//distance between point p and border in direction (O,P) in orger to adapt the width of the wave in the direction OP
    if (o.x>p.x){
        int x=p.x;
        int y=p.y+(o.y-p.y)*(x-p.x)/(o.x-p.x);
        while (true){
            if (!((x-1)>=0 && p.y+(o.y-p.y)*((x-1)-p.x)/(o.x-p.x)>=0 && p.y+(o.y-p.y)*((x-1)-p.x)/(o.x-p.x)<src.rows)) break;
            x--;
            y=p.y+(o.y-p.y)*(x-p.x)/(o.x-p.x);     //we iterate with the equation of the right until reaching  border
        }
        return norm(Point(x,y)-p);
    }
    else if (o.x<p.x){
        int x=p.x;
        int y=int(p.y+(o.y-p.y)*(x-p.x)/(o.x-p.x));
        while (true){
            if (!((x+1)<src.cols && p.y+(o.y-p.y)*((x+1)-p.x)/(o.x-p.x)>=0 && p.y+(o.y-p.y)*((x+1)-p.x)/(o.x-p.x)<src.rows)) break;
            x++;
            y=p.y+(o.y-p.y)*(x-p.x)/(o.x-p.x);
        }
        return norm(Point(x,y)-p);
    }
    else{
        return min(p.y,src.rows-1-p.y);
    }
}

vector< vector<Point> > compressionRadiale(const vector<vector<Point> >& pts, const Point& o, int a, float compr, int largeur){
    vector< vector<Point> > ptsCp(pts);
    for (int i=max(0,o.x-(a+largeur));  i<min(o.x+(a+largeur),src.cols); i++){
        ptsCp[i]=vector<Point>();
    }
    for (int i=max(0,o.x-(a+largeur));  i<min(o.x+(a+largeur),src.cols); i++){
        vector<Point> ptsI=total_ptsX[i];
        for (Point p:ptsI){
            float d=norm(p-o);
            if (d>=a+largeur || d<=a){
                ptsCp[i].push_back(p);
            }
            else{
                int l=distBord(o,p)+d-a+1;      //we modify the width of the wave in the direction (O,P) if the length range extends the limits of the image
                Point q=fComprRadiale(p, o, a, compr, min(largeur,l));//we modify the points of the area of the wave
                ptsCp[q.x].push_back(q);
                is_point_distorted[q.x][q.y]=true;
            }
        }
    }
    return ptsCp;
}


//Circular dilatation effect
Point fDilatRadiale(const Point& z, const Point& o, int R, int largeur){//moves the points from radius [0,R] to [O,R+largeur]
    Point zbis=z-o;
    float r=norm(zbis);
    float rtild=0;
    rtild=(1+float(largeur)/R)*r;
    Point q= o+Point(int(rtild*zbis.x/r),int(rtild*zbis.y/r));
    int x=q.x;
    int y=q.y;
    if (x<0) x=0;
    if (x>=src.cols) x=src.cols-1;
    if (y<0) y=0;
    if (y>=src.rows) y=src.rows-1;
    return Point(x,y);
    
}

vector< vector<Point> > dilatationRadiale(const vector<vector<Point> >& pts, const Point& o, int R, int largeur){
    vector< vector<Point> > ptsDt(pts);
    for (int i=max(0,o.x-(R+largeur));  i<min(o.x+(R+largeur)+1,src.cols); i++){
        ptsDt[i]=vector<Point>();
    }
    for (int i=max(0,o.x-(R+largeur));  i<min(o.x+(R+largeur)+1,src.cols); i++){
        vector<Point> ptsI=pts[i];
        for (Point p:ptsI){
            float d=norm(p-o);
            if (d<=R){
                Point q=fDilatRadiale(p, o, R, largeur);
                ptsDt[q.x].push_back(q);    //we modify the points of the area of the wave
                is_point_distorted[q.x][q.y]=true;
            }
            else if(d>=R+largeur){
                ptsDt[p.x].push_back(p);
            }
        }
    }
    return ptsDt;
}


//Circular compression effect
Point fEcrasmtRadiale(const Point& z, const Point& o, int Ra, int Rb){//moves the points from radius [0,Ra] to [0,Rb] with Rb<Ra
    Point zbis=z-o;
    float r=norm(zbis);
    float rtild=0;
    rtild=float(Rb)*r/Ra;
    Point q= o+Point(int(rtild*zbis.x/r),int(rtild*zbis.y/r));
    int x=q.x;
    int y=q.y;
    if (x<0) x=0;
    if (x>=src.cols) x=src.cols-1;
    if (y<0) y=0;
    if (y>=src.rows) y=src.rows-1;
    return Point(x,y);
}

vector< vector<Point> > ecrasementRadial(const vector<vector<Point> >& pts, const Point& o, int Ra, int Rb){
    vector< vector<Point> > ptsDt(pts);
    for (int i=max(0,o.x-Ra);  i<min(o.x+Ra,src.cols); i++){
        ptsDt[i]=vector<Point>();
    }
    for (int i=max(0,o.x-Ra);  i<min(o.x+Ra+1,src.cols); i++){
        vector<Point> ptsI=pts[i];
        for (Point p:ptsI){
            float d=norm(p-o);
            if (d<=Ra){
                Point q=fEcrasmtRadiale(p, o, Ra, Rb);
                ptsDt[q.x].push_back(q);    //we modify the points of the area of the wave
                is_point_distorted[q.x][q.y]=true;
            }
            else ptsDt[p.x].push_back(p);
        }
    }
    return ptsDt;
}

vector<Point> vectConvexHull(vector< vector <Point> > pts){
    vector<Point>v;
    vector<Point>res;
    for (vector<Point>points:pts){
        for (Point p:points){
            v.push_back(p);
        }
    }
    convexHull(Mat(v), res);
    return res;
}

Point barycentre(const vector<Point>& pts){
    float x=0.,y=0.;
    for (Point p : pts){
        x+=p.x;
        y+=p.y;
    }
    return Point(int(x/pts.size()),int(y/pts.size()));
}

float distMaxPt(const vector<Point>& pts, const Point& o){  //maximum distance between o and Points of pts
    float res=0.;
    for (Point p :pts){
        if (norm(p-o)>res) res=norm(p-o);
    }
    return res;
}

Point randomPointForeground(const vector< vector<Point> >& pts){    //selects randomly a point in the foreground, if it's not too far from the barycenter, otherwise returns the barycenter
    Point res;
    Point bar;
    float dist;
    int numberPoints;
    vector<Point> points;
    for (vector<Point> v : pts){
        for (Point p:v){
            points.push_back(p);
        }
    }
    bar=barycentre(points);
    dist=distMaxPt(points, bar);
    numberPoints=points.size();
    if (numberPoints != 0)
        res= points[rand()%numberPoints];
    else
        return bar;
    if (norm(res-bar)>(4./5)*dist) return bar;//if res is too far
    else return res;
}

Point randomPointMiddle(){    //selects randomly a point in the foreground, if it's not too far from the barycenter, otherwise returns the barycenter
    //srand(time(NULL));
    int a=rand();
    int x=(3*proc_width/8)+(a%(proc_width/4));
    //srand(time(NULL));
    int b=rand();
    int y=(3*proc_height/8)+(b%(proc_height/4));
    return Point(x,y);
}

///SOUND BUFFERS
//Current loudness from buffer
static double computeLoudness(sf::Int16 sBuff[], int n){
    double res=0.;
    for (int i=0; i<n; i++){
        res+=(double)(sBuff[i]*sBuff[i]);
    }
    res*=1./n;
    return res;
}

//Average function
static double avg(const vector<double>& soundBuffer){
    double res = 0.;
    for (double s :soundBuffer) {
        res += s;
    }
    res *=1./ soundBuffer.size();
    return res;
}


//Variance function
static float var(const vector<double>& soundBuffer, double a){
    double res = 0.;
    for (double s :soundBuffer) {
        res += (s - a) * (s - a);
    }
    res *=1./ soundBuffer.size();
    return res;
}

//Update buffer
static void updateBuffer(vector<double>& soundBuffer, double newLoudness){
    for (int i=0; i<soundBuffer.size()-1; i++){
        soundBuffer[i]=soundBuffer[i+1];
    }
    soundBuffer[soundBuffer.size()-1]=newLoudness;
}

static void updateFreqBuffers(vector< vector<double> >& soundBuffer, double newLoudness[]){
    for (int j=0; j<soundBuffer.size(); j++){
        for (int i=0; i<soundBuffer[j].size()-1; i++){
            soundBuffer[j][i]=soundBuffer[j][i+1];
        }
        soundBuffer[j][soundBuffer[j].size()-1]=newLoudness[j];
    }
}



int main()
{
    srand(time(NULL));
    /// SOUND : Initialize input file (to get samples for the Fft)
    sf::InputSoundFile f;
    if (!f.openFromFile(path_track)) {
        cout << "Can't open the file." << endl;
        return -1;
    }
    
    //SOUND DETECTION TOOLS initialization
    //Energy peak detection
    int bSize = 1024;
    sf::Int16 samples[bSize];
    vector<double> previousLoudness(n_intensity_blocks);
    sf::Uint64 count = 1;
    buffer_counter=0;
    double prevAverage, prevVariance;
    mean_buffer_intensity=0.;
    //Fourier oriented detection
    const size_t SIZE = bSize;
    int N=64;
    double freqLoudnesses[N];
    int w1=3;
    double a= 2. * (bSize - N * w1) / (N * (N - 1)); //coefficient échelle logarithmique
    double b= (double)(N * (N + 1) * w1 - 2 * bSize) / (N * (N - 1));   //coefficient échelle logarithmique
    vector< vector<double> > prevFreqLoudness;
    for(int i=0; i<N;i++){
        prevFreqLoudness.push_back(vector<double>(n_intensity_blocks));
    }
    vector<double> prevFreqAvg(N);
    bool isPotentialBeat, isPotentialHit;
    //Time scale
    double bpmT=60./bpm;
    double tLastBeat=0.,tLastHit=0.;
    double tSinceLastBeat=0.,tSinceLastHit=0.;
    
    
    /// SOUND : Initialize music (to hear the track)
    sf::Music music;
    
    if (!music.openFromFile(path_track)) {
        cout << "Can't open the file." << endl;
        return -1;
    }
    
    /// SOUND : Initialize Fft
    auto fft = Aquila::FftFactory::getFft(SIZE);
    
    /// VIDEO : Create windows
    namedWindow( "Delaunay", CV_WINDOW_AUTOSIZE );
    //namedWindow( "Mask", CV_WINDOW_AUTOSIZE );
    
    /// VIDEO : Capture from webcam, set capture size
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;
    cap.set(CV_CAP_PROP_FRAME_WIDTH,disp_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,disp_height);
    
    /// SOUND : Play music
    if (play_music){
        music.play();
        music.setPlayingOffset(sf::seconds(delay_play));
    }
    
    
    /// Create MOG2 Background Subtractor object with no shadows (faster and less errors)
    pMOG2 = createBackgroundSubtractorMOG2(500, 400.0, false);
    
    int key = waitKey(3);
    bool testEffectT;//test if effect at instant t
    
    /// While there are samples to read in the sound file and space bar isn't pressed, play Delaunay with sound & effects
    while (count > 0 && key != 32) {
        /// SOUND : enable/disable sound reactivity by pressing 's'
        if (key == 115) {
            sound_interaction_mode = !sound_interaction_mode;
        }
        
        if (key==3){
            only_beat_mode=!only_beat_mode;
        }
        
        double absBand[SIZE];
        if (sound_interaction_mode) {
            tSinceLastBeat=clock()-tLastBeat;
            tSinceLastHit=clock()-tLastHit;
            isPotentialBeat=false;
            isPotentialHit=false;
            /// SOUND : Grab the samples that are currently played in the music
            f.seek(music.getPlayingOffset());
            count = f.read(samples, bSize);
            
            /// SOUND : Fft on the samples
            double fftSamples[bSize];
            for (int i = 0; i < bSize; i++)
                fftSamples[i] = double(samples[i]);
            const double* fftptr = fftSamples;
            Aquila::SpectrumType spectrum = fft->fft(fftptr);
            for (int i = 0; i < SIZE; i++)
                absBand[i] = abs(spectrum[i]);
        
            float wi;    //largeur de la plage i
            float W = 0;   //indice de la plage au ième passage dans la boucle
            for (int i = 0; i < N; i++) //échelle logarithmique
            {
                float avg = 0;
                wi = a * (i + 1) + b;
                int q = 0;
                for (int j = (int)W; j < (int)(W + wi); j++)
                {
                    avg += absBand[j] * absBand[j];
                    q++;
                }
                freqLoudnesses[i] = avg * q / bSize;
                W += wi;
            }
            updateFreqBuffers(prevFreqLoudness, freqLoudnesses);
            
            /// DETECT RYTHM PATTERNS
            //Detect Beat
            current_sound_intensity=computeLoudness(samples, bSize);
            mean_buffer_intensity+=current_sound_intensity;
            updateBuffer(previousLoudness, current_sound_intensity);
            prevAverage=avg(previousLoudness);
            prevVariance=var(previousLoudness, prevAverage);
            double C = c1 * prevVariance + c2;
            isPotentialBeat=(current_sound_intensity>C*prevAverage && current_sound_intensity>current_to_mean_note*(mean_buffer_intensity/buffer_counter+1));
            
            //Detect Hit
            int nFreqHit=0;
            for (int i=0;i<N;i++){
                prevFreqAvg[i]=avg(prevFreqLoudness[i]);
                if (low_freq_detection_mode){
                    if (i<n_low_frequencies && freqLoudnesses[i]>coef_beat*prevFreqAvg[i]){
                        nFreqHit++;
                    }
                }
                else{
                    if (freqLoudnesses[i]>coef_beat*prevFreqAvg[i])
                        nFreqHit++;
                }
            }
            if (low_freq_detection_mode){
                isPotentialHit=(nFreqHit=n_low_frequencies && current_sound_intensity>current_to_mean_rate_beat*(mean_buffer_intensity/buffer_counter+1));
            }
            else
                isPotentialHit=(nFreqHit>ratio_beat*N && current_sound_intensity>current_to_mean_rate_beat*(mean_buffer_intensity/buffer_counter+1));
            
            //React to rythm through effect release
            if (buffer_counter>n_intensity_blocks && isPotentialHit && (tSinceLastHit/CLOCKS_PER_SEC)>0.95*bpmT){
                key = key_beat_effect;
                tLastHit=clock();
            }
            else if (!only_beat_mode && buffer_counter>n_intensity_blocks && isPotentialBeat && (tSinceLastBeat/CLOCKS_PER_SEC)>1.95*bpmT && (tSinceLastHit/CLOCKS_PER_SEC)>0.5*bpmT){
                key = key_note_effect;
                tLastBeat=clock();
            }
            
            buffer_counter++;
        }
        
        /// BACKGROUND SUBTRACTOR : enable/disable it by pressing 'd'
        if (key == 100) {
            rasterize_background_mode = !rasterize_background_mode;
        }
        
        /// TIME : Initialize time to count Delaunay's execution time
        clock_start = clock();
        
        /// VIDEO : Grab new video frame from camera & reduce size (so that Delaunay is faster)
        cap >> im_src;
        resize(im_src, im_src, Size(proc_width,proc_height));
        
        /// Process image
        DetectEdges();
        
        /// Find good points for Delaunay
        DelaunayPts();
        
        /// EFFECTS : compression effects depending on key pressed
        vector< vector<Point> > dptsT = foreground_ptsX;
        vector< vector<Point> > bgptsT = background_ptsX;
        
        if (key == 2)   wave_front1 = 0;    //Horizontal compression wave if left arrow is pressed
        
        if (key == 3){          //Perpendicular compression wave if right arrow is pressed
            wave_front3 = 0;
            if (rasterize_background_mode){
                wave_Y3=randomPointForeground(total_ptsX).y;   //random ordinate
            }
            else{
                wave_Y3=randomPointForeground(dptsT).y;      //random ordinate in the foreground
            }
            
        }
        
        if (key == 0){              //Circular dilatation effect if up arrow is pressed
            time_wave4 = 0;
            if (rasterize_background_mode){
                //o1=randomPointForeground(ptsX);     //random point in all image
                //o1=Point(dispW/2,dispH/2);
                wave4_origin=randomPointMiddle();
            }
            else{
                conv_hull4=vectConvexHull(dptsT);
                wave4_origin = randomPointForeground(dptsT);      //random point in foreground
            }
            activate_effect[0]=true;
        }
        
        if (key == 13){         //Circular compression wave if enter is pressed
            radial_front = 0;
            if (rasterize_background_mode){
                wave2_origin=randomPointMiddle();     //random point in all image
            }
            else{
                conv_hull2=vectConvexHull(dptsT);
                wave2_origin = randomPointForeground(dptsT);      //random point in foreground
            }
            activate_effect[3]=true;
        }
        
        if (key == 1){          //Circular compression effect if down arrow is pressed
            wave5_time = 0;
            if (rasterize_background_mode){
                //o2=randomPointForeground(ptsX);     //random point in all image
                //o2=Point(dispW/2,dispH/2);
                wave5_origin=randomPointMiddle();
            }
            else{
                conv_hull5=vectConvexHull(dptsT);
                wave5_origin = randomPointForeground(dptsT);      //random point in foreground
            }
            activate_effect[1]=true;
        }
        
        if (key==127){          //Foreground compression effect if backspace is pressed
            wave6_time =0;
            conv_hull6=vectConvexHull(dptsT);
            wave6_origin = barycentre(conv_hull6);//barycenter of the foreground
            wave6_range1=distMaxPt(conv_hull6, wave6_origin);
            activate_effect[2]=true;
        }
        
        //test if effect is present at instant t
        testEffectT=(wave_front1!=-1)||(wave_front3!=-1);
        for (int i=0;i<4;i++){
            testEffectT=testEffectT ||activate_effect[i];
        }
        
        if (testEffectT){
            if (wave_front1 != -1){
                dptsT = CompressHorizontallyImage(dptsT,wave_front1,wave_compr_rate,wave_width1);
                bgptsT = CompressHorizontallyImage(bgptsT,wave_front1,wave_compr_rate,wave_width1);
                wave_front1 += wave_speed1;
                if (wave_front1>=proc_height-1 || (activate_effect[index_beat_effect] && tSinceLastHit<0.45*bpmT) )    //we stop when the wave is completely out of the image or when hit effect has been launched recently
                    wave_front1 = -1;
            }
            
            if (wave_front3 != -1){
                dptsT = translationPerpendiculaire(dptsT, wave_front3, wave_Y3, wave3_param2, wave3_param3, wave3_param1);
                bgptsT = translationPerpendiculaire(bgptsT, wave_front3, wave_Y3, wave3_param2, wave3_param3, wave3_param1);
                wave_front3 += wave_speed3;
                if (wave_front3>=proc_width-1)       //we stop when the wave is completely out of the image
                    wave_front3 = -1;
            }
            if (activate_effect[0]){
                wave4_curr_width = int(wave4_max_width*abs(sin(2*M_PI*wave4_freq*time_wave4)));
                dptsT = dilatationRadiale(dptsT, wave4_origin, wave4_param1, wave4_curr_width);
                bgptsT = dilatationRadiale(bgptsT, wave4_origin, wave4_param1, wave4_curr_width);
                if (rasterize_background_mode==false){
                    for (Point p : conv_hull4){
                        dptsT[p.x].push_back(p);
                    }
                }
                time_wave4++;
                if (time_wave4>1./(2*wave4_freq)){      //we stop when a half period has passed
                    time_wave4 = -1;
                    conv_hull4=vector<Point>();
                    activate_effect[0]=false;
                }
            }
            if (activate_effect[3]){
                dptsT = compressionRadiale(dptsT,wave2_origin,radial_front,wave_compr_rate,wave_width2);
                bgptsT = compressionRadiale(bgptsT,wave2_origin,radial_front,wave_compr_rate,wave_width2);
                if (rasterize_background_mode==false){
                    for (Point p : conv_hull2){
                        dptsT[p.x].push_back(p);      //we keep the hull of the instant of the spark to avoid apparition of holes in the triangulation due to the movement of points
                    }
                }
                radial_front += wave_speed2;
                if (!(wave2_origin.x+radial_front<proc_width-1 || wave2_origin.x-radial_front>0 || wave2_origin.y-radial_front>0 || wave2_origin.y+radial_front<proc_height-1) || (activate_effect[index_beat_effect]&& tSinceLastHit<0.45*bpmT) ){    //we stop when the wave is completely out of the image or when hit effect has been launched recently
                    radial_front = -1;
                    conv_hull2=vector<Point>();
                    activate_effect[3]=false;
                }
            }
            
            if (activate_effect[1]){
                wave5_curr_front = int((1.+13*abs(sin(2*M_PI*wave5_freq *wave5_time)))*wave5_max_front/14);
                dptsT = ecrasementRadial(dptsT, wave5_origin, wave5_max_front, wave5_curr_front);
                bgptsT = ecrasementRadial(bgptsT, wave5_origin, wave5_max_front, wave5_curr_front);
                if (rasterize_background_mode==false){
                    for (Point p : conv_hull5){
                        dptsT[p.x].push_back(p);//we keep the hull of the instant of the spark to avoid apparition of holes in the triangulation due to the movement of points
                    }
                }
                wave5_time++;
                if (wave5_time>1./(2*wave5_freq )){//} || (effectInT[0]&& tSinceLastHit<0.45*bpmT) ){          //we stop when a half period has passed
                    wave5_time=-1;
                    conv_hull5=vector<Point>();
                    activate_effect[1]=false;
                }
            }
            if (activate_effect[2]){
                wave6_range2 = int((1.+15*abs(sin(2*M_PI*wave5_freq *wave6_time )))*wave6_range1/16);
                dptsT = ecrasementRadial(dptsT, wave6_origin, wave6_range1, wave6_range2);
                for (Point p : conv_hull6){
                    dptsT[p.x].push_back(p);//we keep the hull of the instant of the spark to avoid apparition of holes in the triangulation due to the movement of points
                }
                wave6_time ++;
                if (wave6_time >1./(2*wave6_freq)){          //we stop when a half period has passed
                    wave6_time =-1;
                    conv_hull6=vector<Point>();
                    activate_effect[2]=false;
                }
            }
        }
        foreground_ptsX = dptsT;
        background_ptsX=bgptsT;
        
        /// Triangulation
        Triangulate();
        
        /// Colorize the triangles
        ColorizeTriangles();
        
        /// Copy dst to iSrc with dst as mask
        rasterized_image.copyTo(im_src, rasterized_image);
        
        /// Resize to original size and display result
        resize(im_src, im_src, Size(disp_width,disp_height));
        imshow( "Delaunay", im_src);
        
        frame_counter += 1;
        
        /// Exit if space bar is pressed
        key = waitKey(3);
    }
}
