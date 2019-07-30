// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <ctime>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include <ext_list.hpp>
#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "cnn.hpp"
#include "image_grabber.hpp"
#include "text_detection.hpp"
#include "text_recognition.hpp"

#include "text_detection_demo.hpp"

using namespace InferenceEngine;

// ----------------- Variables for Storing and Formatting Text Data from Image ------------------------ //

std::vector<std::string> textData;

std::vector<std::string> keywords{
    "day", "daily","days",
    "hours","hourly","hours",
    // "times","time",
    // "tablet","tablets","capsule","capsules","pill","pills",
    "one","once","1","two","twice","2","three","3","four","4","five","5"
};


// Changes the necessary words to another word
void formatTextData(std::vector<std::string> &textData){

    // Number words to look for
    // Every other element is what the Number word will be converted to
    std::vector<std::string> numberWords{
        "one","1",
        "once","1",
        "two","2",
        "twice","2",
        "three","3",
        "four","4",
        "daily","day"
    };

    // Store new text
    std::vector<std::string> textStore;

    //Add textData text to textStore
    for(auto &i : textData){
        textStore.push_back(i);
    }

    // Look at every other element in the numberWords vector to get the conversion
    for(auto &i : textStore){
        for(int j = 0; j < numberWords.size(); j += 2){
            if(i == numberWords[j]){
                i = numberWords[j + 1];
                break;
            }
        }
    }

    // Clear textData
    textData.clear();

    // Add new text to textData
    for(auto &i : textStore){
        textData.push_back(i);
    }
}

// Gets rid of unnecessary words
// Uses the set data structure to sort the data in some way
void cleanTextData(std::vector<std::string> &textData){

    formatTextData(textData);

    bool check = false;
    std::vector<std::string> textStore;

    /*============ Get all keywords ============== */

    //store all text data into textStore
    for(auto i : textData)
        textStore.push_back(i);

    // Clear text data (will be replaced later)
    textData.clear();

    // Look for key words in the text data from image  
    for(auto &i : textStore){
        for(auto &j : keywords){
             //if a word from the image text is a keyword
            if(i == j){
                check = true;
                break;
            }
        }
        // if a word from the image text is not a key word
        if(!check){
            i = "";
        }
        check = false;
    }

    for(auto &i : textStore){
        if(i != ""){
            textData.push_back(i);
        }
    }

    //clear textStore
    textStore.clear();

    /*============== Clear any repeated words =============== */

    // Sets don't add repeated items
    std::set<std::string> words;

    // Store text data into a set
    for(auto &i : textData){
        words.insert(i);
    }

    std::vector<std::string> wordVect;

    // Add word from set to wordVect
    for(auto &i : words){
        wordVect.push_back(i);
    }

    //Get the 2 most important data points (the number of pills and frequency)
    if(wordVect.size() > 2){
        // start at the end of the vector and get the last 2 things (this should be the number of pills and the frequency)
        for(int i = 2; i < wordVect.size(); i++){
            wordVect[wordVect.size() - i - 1] = "";
        }
    }

    // Clear textData
    textData.clear();

    // Add words to textData (words should not repeat)
    for(auto &i : wordVect){
        if(i != "")
            textData.push_back(i);
    }

    /*
    // Look for key words in the text data from image
    for(int i = 0; i < textData.size(); i++){
        for(int j = 0; j < keywords.size(); j++){
            //if a word from the image text is a keyword
            if(textData[i] == keywords[j]){
                check = true;
                break;
            }
        }
        //if a word from the image text is not a key word
        if(!check){
           textData.erase(textData.begin() + i); 
        }
        check = false;
    }
    */
}

//Finalizes data in the format needed
void finalizeTextData(std::vector<std::string> &textData){

    // Initialized vector to 3 blank elements
    std::vector<std::string> finalText(3, "");
    /*
        Vector format:

        { directions (ex. " 2 hours "), medicine name (ex. "ibuprofen"), start date (ex. "7/24/2019")}
     */

    // Cleans text data and gets rid of unnecessary words
    cleanTextData(textData);

    // 1. Fill in the directions
    finalText[0] = "Take " + textData[0] + " pills every " + textData[1]; 

    // 2. Fill in the medicine name
    

    finalText[1] = "";

    // 3. Fill in date

    finalText[2] = "";

    // Clear text data
    textData.clear();

    for(auto &i : finalText){
        textData.push_back(i);
    }

}

// ---------------------------------------------------------------------------------------------------//
std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect);
std::vector<cv::Point> boundedIntPointsFromRotatedRect(const cv::RotatedRect &rect, const cv::Size& image_size);
cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx);
cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx);
void setLabel(cv::Mat& im, const std::string label, const cv::Point & p);

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ------------------------- Parsing and validating input arguments --------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_td.empty() && FLAGS_m_tr.empty()) {
        throw std::logic_error("Neither parameter -m_td nor -m_tr is not set");
    }

    return true;
}

int clip(int x, int max_val) {
    return std::min(std::max(x, 0), max_val);
}

int main(int argc, char *argv[]) {
    try {
        // ----------------------------- Parsing and validating input arguments ------------------------------

        /** This demo covers one certain topology and cannot be generalized **/

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        double text_detection_postproc_time = 0;
        double text_recognition_postproc_time = 0;
        double text_crop_time = 0;
        double avg_time = 0;
        const double avg_time_decay = 0.8;

        const char kPadSymbol = '#';
        std::string kAlphabet = std::string("0123456789abcdefghijklmnopqrstuvwxyz") + kPadSymbol;

        const double min_text_recognition_confidence = FLAGS_thr;

        std::map<std::string, InferencePlugin> plugins_for_devices;
        std::vector<std::string> devices = {FLAGS_d_td, FLAGS_d_tr};

        float cls_conf_threshold = static_cast<float>(FLAGS_cls_pixel_thr);
        float link_conf_threshold = static_cast<float>(FLAGS_link_pixel_thr);

        for (const auto &device : devices) {
            if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
                continue;
            }
            InferencePlugin plugin = PluginDispatcher().getPluginByDevice(device);
            /** Load extensions for the CPU plugin **/
            if ((device.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            plugins_for_devices[device] = plugin;
        }

        auto image_path = FLAGS_i;
        auto text_detection_model_path = FLAGS_m_td;
        auto text_recognition_model_path = FLAGS_m_tr;
        auto extension_path = FLAGS_l;

        Cnn text_detection, text_recognition;

        if (!FLAGS_m_td.empty())
            text_detection.Init(FLAGS_m_td, &plugins_for_devices[FLAGS_d_td], cv::Size(FLAGS_w_td, FLAGS_h_td));

        if (!FLAGS_m_tr.empty())
            text_recognition.Init(FLAGS_m_tr, &plugins_for_devices[FLAGS_d_tr]);

        std::unique_ptr<Grabber> grabber = Grabber::make_grabber(FLAGS_dt, FLAGS_i);

        cv::Mat image;
        grabber->GrabNextImage(&image);

        std::cout << "Words Detected: ";

        while (!image.empty()) {
            cv::Mat demo_image = image.clone();
            cv::Size orig_image_size = image.size();

            std::chrono::steady_clock::time_point begin_frame = std::chrono::steady_clock::now();
            std::vector<cv::RotatedRect> rects;
            if (text_detection.is_initialized()) {
                auto blobs = text_detection.Infer(image);
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                rects = postProcess(blobs, orig_image_size, cls_conf_threshold, link_conf_threshold);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                text_detection_postproc_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            } else {
                rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f);
            }

            if (FLAGS_max_rect_num >= 0 && static_cast<int>(rects.size()) > FLAGS_max_rect_num) {
                std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect & a, const cv::RotatedRect & b) {
                    return a.size.area() > b.size.area();
                });
                rects.resize(FLAGS_max_rect_num);
            }

            int num_found = text_recognition.is_initialized() ? 0 : static_cast<int>(rects.size());

            for (const auto &rect : rects) {
                cv::Mat cropped_text;
                std::vector<cv::Point2f> points;
                int top_left_point_idx = -1;

                if (rect.size != cv::Size2f(0, 0) && text_detection.is_initialized()) {
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    points = floatPointsFromRotatedRect(rect);
                    topLeftPoint(points, &top_left_point_idx);
                    cropped_text = cropImage(image, points, text_recognition.input_size(), top_left_point_idx);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_crop_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                } else {
                    cropped_text = image;
                }

                /* =============== This is where text is printed out ================== */

                std::string res = "";
                double conf = 1.0;
                if (text_recognition.is_initialized()) {
                    auto blobs = text_recognition.Infer(cropped_text);
                    auto output_shape = blobs.begin()->second->getTensorDesc().getDims();
                    if (output_shape[2] != kAlphabet.length())
                        throw std::runtime_error("The text recognition model does not correspond to alphabet.");

                    float *ouput_data_pointer = blobs.begin()->second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                    std::vector<float> output_data(ouput_data_pointer, ouput_data_pointer + output_shape[0] * output_shape[2]);

                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    res = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_recognition_postproc_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

                    res = conf >= min_text_recognition_confidence ? res : "";
                    num_found += !res.empty() ? 1 : 0;
                }

                if (FLAGS_r) {
                    /* 
                    for (size_t i = 0; i < points.size(); i++) {
                        std::cout << "(" << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
                                     clip(static_cast<int>(points[i].y), image.rows - 1) << ")";
                        if (i != points.size() - 1)
                            std::cout << ",";
                    }
                    */
                    
                     
                     if (text_recognition.is_initialized()) {
                         std::cout << res << ", ";
                      }
                    

                    textData.push_back(res);

                    // std::cout << std::endl;

                    // std::cout << res;
                }

                if (!FLAGS_no_show && (!res.empty() || !text_recognition.is_initialized())) {
                    for (size_t i = 0; i < points.size() ; i++) {
                        cv::line(demo_image, points[i], points[(i+1) % points.size()], cv::Scalar(50, 205, 50), 2);
                    }

                    if (!points.empty() && !res.empty()) {
                        setLabel(demo_image, res, points[top_left_point_idx]);
                    }
                }
            }

            std::chrono::steady_clock::time_point end_frame = std::chrono::steady_clock::now();

            if (avg_time == 0) {
                avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count());
            } else {
                auto cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count();
                avg_time = avg_time * avg_time_decay + (1.0 - avg_time_decay) * cur_time;
            }
            int fps = static_cast<int>(1000 / avg_time);

            if (!FLAGS_no_show) {
               // std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
                cv::putText(demo_image, "fps: " + std::to_string(fps) + " found: " + std::to_string(num_found),
                            cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
               // cv::imshow("Press any key to exit", demo_image);
                char k = cv::waitKey(3);
                if (k == 27) break;
            }

            grabber->GrabNextImage(&image);
        }

        // Get rid of any uonnecessary words that aren't keywords
        // Format text
        finalizeTextData(textData);

        // Print out text captured from image
        std::cout << std::endl;
        std::cout << "Directions: " << textData[0] << std::endl;
        // std::cout << "Medicine Name: " << textData[1] << std::endl;
        // std::cout << "Start Date: " << textData[2] << std::endl;

        if (text_detection.ncalls() && !FLAGS_r) {
          std::cout << "text detection model inference (ms) (fps): "
                    << text_detection.time_elapsed() / text_detection.ncalls() << " "
                    << text_detection.ncalls() * 1000 / text_detection.time_elapsed() << std::endl;
        if (std::fabs(text_detection_postproc_time) < std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("text_detection_postproc_time can't be equal to zero");
        }
          std::cout << "text detection postprocessing (ms) (fps): "
                    << text_detection_postproc_time / text_detection.ncalls() << " "
                    << text_detection.ncalls() * 1000 / text_detection_postproc_time << std::endl << std::endl;
        }

        if (text_recognition.ncalls() && !FLAGS_r) {
          std::cout << "text recognition model inference (ms) (fps): "
                    << text_recognition.time_elapsed() / text_recognition.ncalls() << " "
                    << text_recognition.ncalls() * 1000 / text_recognition.time_elapsed() << std::endl;
          if (std::fabs(text_recognition_postproc_time) < std::numeric_limits<double>::epsilon()) {
              throw std::logic_error("text_recognition_postproc_time can't be equal to zero");
          }
          std::cout << "text recognition postprocessing (ms) (fps): "
                    << text_recognition_postproc_time / text_recognition.ncalls() / 1000 << " "
                    << text_recognition.ncalls() * 1000000 / text_recognition_postproc_time << std::endl << std::endl;
          if (std::fabs(text_crop_time) < std::numeric_limits<double>::epsilon()) {
              throw std::logic_error("text_crop_time can't be equal to zero");
          }
          std::cout << "text crop (ms) (fps): " << text_crop_time / text_recognition.ncalls() / 1000 << " "
                    << text_recognition.ncalls() * 1000000 / text_crop_time << std::endl << std::endl;
        }

        // ---------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; i++) {
        points.emplace_back(vertices[i].x, vertices[i].y);
    }
    return points;
}

cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx) {
    cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    int most_left_idx = -1;
    int almost_most_left_idx = -1;

    for (size_t i = 0; i < points.size() ; i++) {
        if (most_left.x > points[i].x) {
            if (most_left.x != std::numeric_limits<float>::max()) {
                almost_most_left = most_left;
                almost_most_left_idx = most_left_idx;
            }
            most_left = points[i];
            most_left_idx = i;
        }
        if (almost_most_left.x > points[i].x && points[i] != most_left) {
            almost_most_left = points[i];
            almost_most_left_idx = i;
        }
    }

    if (almost_most_left.y < most_left.y) {
        most_left = almost_most_left;
        most_left_idx = almost_most_left_idx;
    }

    *idx = most_left_idx;
    return most_left;
}

cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx) {
    cv::Point2f point0 = points[top_left_point_idx];
    cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
    cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

    cv::Mat crop(target_size, CV_8UC3, cv::Scalar(0));

    std::vector<cv::Point2f> from{point0, point1, point2};
    std::vector<cv::Point2f> to{cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(target_size.width-1), 0.0f),
                                cv::Point2f(static_cast<float>(target_size.width-1), static_cast<float>(target_size.height-1))};

    cv::Mat M = cv::getAffineTransform(from, to);

    cv::warpAffine(image, crop, M, crop.size());

    return crop;
}

void setLabel(cv::Mat& im, const std::string label, const cv::Point & p) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.7;
    int thickness = 1;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    auto text_position = p;
    text_position.x = std::max(0, p.x);
    text_position.y = std::max(text_size.height, p.y);

    cv::rectangle(im, text_position + cv::Point(0, baseline), text_position + cv::Point(text_size.width, -text_size.height), CV_RGB(50, 205, 50), cv::FILLED);
    cv::putText(im, label, text_position, fontface, scale, CV_RGB(255, 255, 255), thickness, 8);
}
