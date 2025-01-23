#include "AutoLabelingToolImpl.h"

//#include "cuda_provider_factory.h"
#include <thread>

#include "util.h"

std::tuple<int, int> GetPreProcessShape(int old_h, int old_w, int long_side_length)
{
	/*
		Compute the output size given input size and target long side length.
	*/
	double scale = long_side_length * 1.0 / MAX(old_h, old_w);
	int new_h = (int)(old_h * scale + 0.5);
	int new_w = (int)(old_w * scale + 0.5);
	std::tuple<int, int> newShape(new_h, new_w);
	return newShape;
}

//Ort::Env AutoLabelingTool::AutoLabelingToolImpl::env = nullptr;
AutoLabelingTool::AutoLabelingToolImpl::AutoLabelingToolImpl()
{
	m_bModelLoaded = false;
	imageEmbeddingValue = nullptr;
}

AutoLabelingTool::AutoLabelingToolImpl::~AutoLabelingToolImpl()
{
	if (imageEmbeddingValue != nullptr)
	{
		delete[] imageEmbeddingValue;
		imageEmbeddingValue = nullptr;
	}
	
}

bool AutoLabelingTool::AutoLabelingToolImpl::InitModel(const std::wstring &strModelPath, const int nDeviceMode)
{
	try
	{
		m_bModelLoaded = false;

		if (!outputTensorValuesPre.empty())
		{
			std::vector<float>().swap(outputTensorValuesPre);
		}
		if (!inputTensorValuesPre.empty())
		{
			std::vector<float>().swap(inputTensorValuesPre);
		}

		if (!env) {
			env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_SAM");
		}

		/*std::vector<std::string> availableProviders = Ort::GetAvailableProviders();

		for (size_t ii = 0; ii < availableProviders.size(); ii++)
		{
			LOGPRINTF(stderr, "%s\n", availableProviders[ii].c_str());
		}*/

		sessionOptions = Ort::SessionOptions();
		sessionOptions.SetIntraOpNumThreads((int)(std::thread::hardware_concurrency() * 0.75));
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		if (nDeviceMode == 1)//GPU
		{
			//m_Session = new Ort::Session(m_Env, modelPath.c_str(), Ort::SessionOptions{ nullptr });
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));//gpu_id : 0
			/*const auto& api = Ort::GetApi();
			OrtTensorRTProviderOptionsV2* tensorrt_options;

			Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
			std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
				tensorrt_options, api.ReleaseTensorRTProviderOptions);
			Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(sessionOptions),
				rel_trt_options.get()));*/
		}

		std::wstring strPreModelPath = strModelPath + L"\\ENCODER.onnx";
		sessionPre = Ort::Session(env, strPreModelPath.c_str(), sessionOptions);
		std::wstring strPointModelPath = strModelPath + L"\\DECODER_POINT.onnx";
		sessionPoint = Ort::Session(env, strPointModelPath.c_str(), sessionOptions);

		//Preprocessing model
		inputShapePre = { m_nBatchSize, 3, 1024, 1024 };
		outputShapePre = { m_nBatchSize, 256, 64, 64 };

		inputTensorValuesPre.resize(inputShapePre[0] * inputShapePre[1] * inputShapePre[2] * inputShapePre[3]);
		outputTensorValuesPre.resize(outputShapePre[0] * outputShapePre[1] * outputShapePre[2] * outputShapePre[3]);
		input_tensorPre = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValuesPre.data(), inputTensorValuesPre.size(),
			inputShapePre.data(), inputShapePre.size());
		output_tensorPre = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValuesPre.data(), outputTensorValuesPre.size(),
			outputShapePre.data(), outputShapePre.size());

		if (imageEmbeddingValue == nullptr)
		{
			imageEmbeddingValue = new float[1048576];
			if (imageEmbeddingValue == nullptr)
			{
				return false;
			}
		}
		

		m_bModelLoaded = true;
	}
	catch (std::exception &err)
	{
		return false;
	}
	catch (...)
	{
		return false;
	}


	return true;
}

bool AutoLabelingTool::AutoLabelingToolImpl::PreProcess(std::vector<std::vector<float>> &vvImageCache, const std::vector<std::wstring> &vImagePath)
{
	if (!m_bModelLoaded)
	{
		return false;
	}

	if (vImagePath.size() == 0)
	{
		return false;
	}

	try
	{
		if (!vvImageCache.empty())
		{
			std::vector<std::vector<float>>().swap(vvImageCache);
		}

		int nImageNum = vImagePath.size();
		vvImageCache.resize(nImageNum);

		int nBatchNum = 0;
		int nBatchSum = 0;
		for (int i = 0; i < nImageNum; i++)
		{
			cv::Mat img = cv::imread(Ws2S(vImagePath[i]), -1);

			if (!img.data)
			{
				return false;
			}

			if (img.channels() == 1)
			{
				cvtColor(img, img, cv::COLOR_GRAY2RGB);
			}
			else
			{
				cvtColor(img, img, cv::COLOR_BGR2RGB);//BGR->RGB
			}

			if (!PreProcessImage(img, nBatchNum, m_nBatchSize))
			{
				return false;
			}

			nBatchNum++;
			if (nBatchNum == m_nBatchSize)
			{
				Ort::RunOptions run_options;
				sessionPre.Run(run_options, inputNamesPre, &input_tensorPre, 1, outputNamesPre, &output_tensorPre, 1);

				vvImageCache[i] = outputTensorValuesPre;

				for (int ii = 0; ii < 10; ii++)
				{
					LOGPRINTF(stderr, "input values %d : %f\n", ii, inputTensorValuesPre[ii]);
					LOGPRINTF(stderr, "output values %d : %f\n", ii, outputTensorValuesPre[ii]);
				}

				nBatchSum += m_nBatchSize;
				nBatchNum = 0;
			}


		}
	}
	catch (std::exception &err)
	{
		std::string str = err.what();
		return false;
	}
	catch (...)
	{
		return false;
	}

	return true;
}

bool AutoLabelingTool::AutoLabelingToolImpl::Execute(std::vector<float> &vMaskCache,
	const std::vector<float> &vImageCache, const std::vector<LabelPoint> &pts, const std::pair<int, int> &OriImgSize,
	const std::vector<float> &vLastMaskCache /*= {}*/)
{
	if (!m_bModelLoaded)
	{
		return false;
	}

	if (vImageCache.size() != 1048576)// 1x256x64x64
	{
		return false;
	}

	if (OriImgSize.second <= 0 || OriImgSize.first <= 0 || pts.size() == 0)
	{
		return false;
	}

	if (pts[0].m_nType != 1)// the first point must be positive
	{
		return false;
	}

	for (size_t i = 0; i < pts.size(); i++)
	{
		if (pts[i].m_nX < 0 || pts[i].m_nX >= OriImgSize.first || pts[i].m_nY < 0 || pts[i].m_nY >= OriImgSize.second ||
			pts[i].m_nType < 0 || pts[i].m_nType > 1)
		{
			return false;
		}
	}

	if (vLastMaskCache.size() != 0 && vLastMaskCache.size() != 65536)//256x256
	{
		return false;
	}

	if (!vMaskCache.empty())
	{
		std::vector<float>().swap(vMaskCache);
	}
	vMaskCache.resize(65536);

	if (!ExecuteSAM(vMaskCache, vImageCache, pts, OriImgSize, vLastMaskCache))
	{
		return false;
	}

	return true;
}

void prepare_tensor(const std::vector<float> &inputValues, float*& blob)
{
	size_t nSize = inputValues.size();
	blob = new float[nSize];

	for (size_t i = 0; i < nSize; i++)
	{
		blob[i] = inputValues[i];
	}
}

bool AutoLabelingTool::AutoLabelingToolImpl::ExecuteSAM(std::vector<float> &vMaskCache, const std::vector<float> &vImageCache,
	const std::vector<LabelPoint> &pts, const std::pair<int, int> &OriImgSize, const std::vector<float> &vLastMaskCache /*= {}*/)
{
	std::tuple<int, int> newShape = GetPreProcessShape(OriImgSize.second, OriImgSize.first, long_side_length);

	float new_w = static_cast<float>(std::get<1>(newShape));
	float new_h = static_cast<float>(std::get<0>(newShape));

	float ratio_x = (new_w / OriImgSize.first);
	float ratio_y = (new_h / OriImgSize.second);

	std::vector<float> inputPointValues, inputLabelValues;

	bool bRect = false;
	
	for (size_t i = 0; i < pts.size(); i++)
	{
		float new_x = pts[i].m_nX * ratio_x;
		float new_y = pts[i].m_nY * ratio_y;
		if (pts[i].m_nType == 0)//negative
		{
			inputPointValues.push_back(new_x);
			inputPointValues.push_back(new_y);
			inputLabelValues.push_back(0);
		}
		else//positive
		{
			inputPointValues.push_back(new_x);
			inputPointValues.push_back(new_y);
			inputLabelValues.push_back(1);
		}
	}

	if (!bRect)
	{
		inputPointValues.push_back(0.0f);
		inputPointValues.push_back(0.0f);
		inputLabelValues.push_back(-1);
	}

	const int numPoints = inputLabelValues.size();
	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 }, pointLabelsShape = { 1, numPoints },
		maskInputShape = { 1, 1, 256, 256 }, hasMaskInputShape = { 1 },
		origImSizeShape = { 2 }, inputEmbeddingShape = { 1, 256, 64, 64 };

	std::vector<Ort::Value> inputTensorsSam;

	float* blob_x = nullptr;
	float* blob_y = nullptr;
	
	std::transform(vImageCache.begin(), vImageCache.end(), imageEmbeddingValue, [](const float x)
	{
		return x;
	});

	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, imageEmbeddingValue, 1048576,
		inputEmbeddingShape.data(), inputEmbeddingShape.size()));

	for (int ii = 0; ii < 10; ii++)
	{
		LOGPRINTF(stderr, "input SAM values %d : %f\n", ii, imageEmbeddingValue[ii]);
	}


	prepare_tensor(inputPointValues, blob_x);
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, blob_x,
		2 * numPoints, inputPointShape.data(),
		inputPointShape.size()));

	for (int ii = 0; ii < inputPointValues.size(); ii++)
	{
		LOGPRINTF(stderr, "input point values %d : %f\n", ii, blob_x[ii]);
	}
	prepare_tensor(inputLabelValues, blob_y);
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, blob_y,
		numPoints, pointLabelsShape.data(),
		pointLabelsShape.size()));

	for (int ii = 0; ii < inputLabelValues.size(); ii++)
	{
		LOGPRINTF(stderr, "input label values %d : %f\n", ii, blob_y[ii]);
	}

	const size_t maskInputSize = 256 * 256;
	float maskInputValues[maskInputSize],
		hasMaskValues[] = { 0 },
		orig_im_size_values[] = { (float)OriImgSize.second, (float)OriImgSize.first };

	if (vLastMaskCache.size() == 0)
	{
		memset(maskInputValues, 0, maskInputSize);
	}
	else
	{
		hasMaskValues[0] = 1;
		std::transform(vLastMaskCache.begin(), vLastMaskCache.end(), maskInputValues, [](const float x)
		{
			return x;
		});
	}

	/*for (int ii = 0; ii < 10; ii++)
	{
		LOGPRINTF(stderr, "input mask values %d : %f\n", ii, maskInputValues[ii]);
	}*/

	inputTensorsSam.push_back(
		Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize,
			maskInputShape.data(), maskInputShape.size()));

	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));

	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

	/*LOGPRINTF(stderr, "hasMaskValues : %d\n", hasMaskValues[0]);

	for (int ii = 0; ii < 2; ii++)
	{
		LOGPRINTF(stderr, "orig_im_size_values %d : %f\n", ii, orig_im_size_values[ii]);
	}*/

	for (int ii = 0; ii < inputTensorsSam.size(); ii++)
	{
		auto &input = inputTensorsSam[ii];
		size_t nCount = input.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
		LOGPRINTF(stderr, "input num values : %d\n", nCount);

		if (nCount > 10)
		{
			nCount = 10;
		}

		auto value = inputTensorsSam[ii].GetTensorMutableData<float>();

		for (size_t jj = 0; jj < nCount; jj++)
		{
			LOGPRINTF(stderr, "input values %d : %f\n", jj, value[jj]);
		}
	}


	Ort::RunOptions runOptionsSam;
	std::vector<Ort::Value> outputTensorsSam;
	if (bRect)//Box
	{
		
	}
	else//Point
	{
		outputTensorsSam = sessionPoint.Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(),
			inputTensorsSam.size(), outputNamesSam, 3);

		//sessionPoint.Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), 6, outputNamesSam, outputTensorsSam.data(), 3);
	}

	if (outputTensorsSam.size() != 3)
	{
		return false;
	}

	auto masks = outputTensorsSam[0].GetTensorMutableData<float>();
	auto iou_predictions = outputTensorsSam[1].GetTensorMutableData<float>();
	auto low_res_masks = outputTensorsSam[2].GetTensorMutableData<float>();

	Ort::Value& masks_ = outputTensorsSam[0];
	Ort::Value& iou_predictions_ = outputTensorsSam[1];
	Ort::Value& low_res_masks_ = outputTensorsSam[2];

	auto mask_dims = masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto iou_pred_dims = iou_predictions_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto low_res_dims = low_res_masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();

	for (int ii = 0; ii < 10; ii++)
	{
		LOGPRINTF(stderr, "mask output values %d : %f\n", ii, masks[ii]);
	}

	for (int ii = 0; ii < mask_dims.size(); ii++)
	{
		LOGPRINTF(stderr, "mask shape values %d : %d\n", ii, mask_dims[ii]);
	}

	for (int ii = 0; ii < iou_pred_dims.size(); ii++)
	{
		LOGPRINTF(stderr, "iou shape values %d : %d\n", ii, iou_pred_dims[ii]);
	}

	for (int ii = 0; ii < low_res_dims.size(); ii++)
	{
		LOGPRINTF(stderr, "low shape values %d : %d\n", ii, low_res_dims[ii]);
	}

	LOGPRINTF(stderr, "mask num values : %d\n", masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount());

	int masks_nums = static_cast<int>(mask_dims.at(1));
	float maxValue = -1;
	int maxID = -1;
	for (int i = 0; i < masks_nums; i++)
	{
		float iouValue = *(iou_predictions++);
		if (iouValue > maxValue)
		{
			maxValue = iouValue;
			maxID = i;
		}
	}

	// save cache
	int nJumpRate = maxID * 65536;
	memcpy(&(vMaskCache[0]), low_res_masks + nJumpRate, 65536 * sizeof(float));

	nJumpRate = maxID * mask_dims.at(2) * mask_dims.at(3);

	// save mask
	/*if (label_result.Height() == OriImgSize.second && label_result.Width() == OriImgSize.first)
	{
		for (int i = 0; i < label_result.Height(); i++)
		{
			for (int j = 0; j < label_result.Width(); j++)
			{
				uchar value = masks[i * label_result.Width() + j + nJumpRate] > 0 ? 0 : 255;
				label_result.SetPixelValue(j, i, value);
			}
		}
	}
	else*/
	{
		cv::Mat mask = cv::Mat(1200, 1800, CV_8UC1, cv::Scalar(255));
		if (!mask.data)
		{
			return false;
		}
		unsigned char* ptr = mask.ptr<unsigned char>(0);
		for (int i = 0; i < mask.rows; i++)
		{
			for (int j = 0; j < mask.cols; j++)
			{
				if (masks[i * mask.cols + j + nJumpRate] > 0)
				{
					ptr[i * mask.cols + j] = 0;
				}
			}
		}

		//cv::imwrite("C:\\result.bmp", mask);
	}

	delete[] blob_x;
	blob_x = nullptr;

	delete[] blob_y;
	blob_y = nullptr;

	return true;
}

bool AutoLabelingTool::AutoLabelingToolImpl::PreProcessImage(const cv::Mat &imgInput, const int nBatchNum, const int nBatchSize)
{
	const unsigned int h = imgInput.rows;
	const unsigned int w = imgInput.cols;

	std::tuple<int, int> newShape = GetPreProcessShape(h, w, long_side_length);

	cv::Mat imgResize;
	resize(imgInput, imgResize, cv::Size(std::get<1>(newShape), std::get<0>(newShape)), cv::INTER_LINEAR);

	std::vector<float> m_vMean = { 123.675, 116.28, 103.53 };
	std::vector<float> m_vStd = { 58.395, 57.12, 57.375 };

	const cv::Mat imgSampleFloat(long_side_length, long_side_length, CV_32FC3, cv::Scalar(m_vMean[0], m_vMean[1], m_vMean[2]));
	imgResize.copyTo(imgSampleFloat(cv::Rect(0, 0, imgResize.cols, imgResize.rows)));
	
	//RGBRGBRGB..... to RRRRRRGGGGGBBBB......
	const int nChannel = imgSampleFloat.channels();
	const int nSize = imgSampleFloat.cols * imgSampleFloat.rows * nChannel;

	int k = -1;
	const int nImageStart = nBatchNum * nSize;
	const float* ptr = imgSampleFloat.ptr<float>(0);
	for (uint32_t c = 0; c < nChannel; c++)
	{
		float fMean = m_vMean[c];
		float fStd = m_vStd[c];
		for (uint32_t p = c; p < nSize; p = p + nChannel)
		{
			inputTensorValuesPre[++k + nImageStart] = (float(*(ptr + p)) - fMean) / fStd;
		}
	}

	return true;
}