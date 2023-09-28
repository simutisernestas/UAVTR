#include <gtest/gtest.h>
#include "tracker.h"

TEST(TrackerTest, TestTracker)
{
    // Create a tracker object
    Tracker tracker;

    // Create a sequence of frames to track
    std::vector<cv::Mat> frames;
    frames.push_back(cv::imread("frame1.jpg"));
    frames.push_back(cv::imread("frame2.jpg"));
    frames.push_back(cv::imread("frame3.jpg"));

    // Set up the tracker with the first frame and a bounding box
    cv::Rect bbox(100, 100, 50, 50);
    tracker.init(frames[0], bbox);

    // Update the tracker with the remaining frames
    for (int i = 1; i < frames.size(); i++)
    {
        bool track = tracker.update(frames[i], bbox);
        ASSERT_TRUE(track);
    }
}

TEST(PreprocessTest, ConvertsBGRToRGB)
{
    // Arrange
    cv::Mat image = cv::imread("test_image.jpg");
    cv::Mat expected = image.clone();
    cv::cvtColor(expected, expected, cv::COLOR_BGR2RGB);

    // Act
    cv::Mat result = preprocess(image);

    // Assert
    ASSERT_EQ(result.size(), expected.size());
    ASSERT_EQ(result.type(), expected.type());
    ASSERT_TRUE(cv::countNonZero(result != expected) == 0);
}

TEST(PreprocessTest, ResizesImage)
{
    // Arrange
    cv::Mat image = cv::imread("test_image.jpg");
    cv::Size expectedSize(224, 224);

    // Act
    cv::Mat result = preprocess(image);

    // Assert
    ASSERT_EQ(result.size(), expectedSize);
}

TEST(PreprocessTest, ConvertsToFloat32AndNormalizes)
{
    // Arrange
    cv::Mat image = cv::imread("test_image.jpg");

    // Act
    cv::Mat result = preprocess(image);

    // Assert
    ASSERT_EQ(result.type(), CV_32F);
    ASSERT_TRUE(cv::countNonZero(result < 0) == 0);
    ASSERT_TRUE(cv::countNonZero(result > 1) == 0);
}

TEST(PreprocessTest, Creates4DBlob)
{
    // Arrange
    cv::Mat image = cv::imread("test_image.jpg");

    // Act
    cv::Mat result = preprocess(image);

    // Assert
    ASSERT_EQ(result.dims, 4);
    ASSERT_EQ(result.size[0], 1);
    ASSERT_EQ(result.size[1], 3);
    ASSERT_EQ(result.size[2], 224);
    ASSERT_EQ(result.size[3], 224);
}