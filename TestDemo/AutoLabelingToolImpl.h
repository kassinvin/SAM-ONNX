#pragma once
#include "AutoLabelingTool.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#pragma pack(push,8)

class AutoLabelingTool::AutoLabelingToolImpl
{
public:
	AutoLabelingToolImpl();

	~AutoLabelingToolImpl();

	bool InitModel(const std::wstring &strModelPath, const int nDeviceMode);

	bool PreProcess(std::vector<std::vector<float>> &vvImageCache, const std::vector<std::wstring> &vImagePath);

	bool Execute(std::vector<float> &vMaskCache, 
		const std::vector<float> &vImageCache, const std::vector<LabelPoint> &pts, const std::pair<int, int> &OriImgSize,
		const std::vector<float> &vLastMaskCache = {});

private:
	/*static */Ort::Env env{ nullptr };
	Ort::SessionOptions sessionOptions{ nullptr };
	Ort::Session sessionPre{ nullptr }, sessionPoint{ nullptr };

	Ort::MemoryInfo memoryInfo{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };

	const char *inputNamesSam[6]{ "image_embeddings", "point_coords",   "point_labels",
						 "mask_input",       "has_mask_input", "orig_im_size" },
		*outputNamesSam[3]{ "masks", "iou_predictions", "low_res_masks" },
		*inputNamesPre[1]{ "input_image" }, 
		*outputNamesPre[1]{ "image_embeddings" };

	bool m_bModelLoaded;
	const int long_side_length = 1024;
	const int m_nBatchSize = 1;

	std::vector<int64_t> inputShapePre, outputShapePre;
	std::vector<float> outputTensorValuesPre, inputTensorValuesPre;
	Ort::Value input_tensorPre{ nullptr }, output_tensorPre{ nullptr };

	float *imageEmbeddingValue;

	bool PreProcessImage(const cv::Mat &imgInput, const int nBatchNum, const int nBatchSize);

	bool ExecuteSAM(std::vector<float> &vMaskCache, const std::vector<float> &vImageCache, 
		const std::vector<LabelPoint> &pts, const std::pair<int, int> &OriImgSize, const std::vector<float> &vLastMaskCache = {});

};

#pragma pack(pop)
