#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <memory>
#include <string>
#include <stdexcept>
#include <chrono>

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args ){
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

const std::string BASE_DIR = "/home/marko/CLionProjects/PROJEKT/";
const std::string OBJ_DIR = BASE_DIR + "objects/";

int main (){
    std::string input_file = OBJ_DIR + "Vinograd_mali-Cloud.ply";
    std::string output_file = OBJ_DIR + "Vinograd_mali-Cloud_%s-k=%d-sigma=%.1f.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    pcl::PLYReader reader;
    // Replace the path below with the path where you saved your file
    reader.read<pcl::PointXYZ> (input_file, *cloud);

    std::cerr << "Cloud before filtering: " << std::endl;
    std::cerr << *cloud << std::endl;

    // Create the filtering object
    using namespace std::chrono;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    auto nrK = 50;
    sor.setMeanK (nrK);
    auto stdevMultl = 1.0;
    sor.setStddevMulThresh (stdevMultl);
    auto start = high_resolution_clock::now();
    sor.filter (*cloud_filtered);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    std::cout << duration.count() << std::endl;

    std::cerr << "Cloud after filtering: " << std::endl;
    std::cerr << *cloud_filtered << std::endl;

    pcl::PLYWriter writer;
    auto output = string_format(output_file, "inliers", nrK, stdevMultl);
    writer.write<pcl::PointXYZ> (output, *cloud_filtered, false);

    sor.setNegative (true);
    sor.filter (*cloud_filtered);

    writer.write<pcl::PointXYZ> (string_format(output_file, "outliers", nrK, stdevMultl), *cloud_filtered, false);

    return (0);
}

