#pragma once

#pragma pack(push,8)
#include "util.h"

class AutoLabelingTool
{
public:
	class AutoLabelingToolImpl;

public:
	AutoLabelingTool();

	~AutoLabelingTool();

	/*================================================
	input:
	@ strModelPath:
	@ nDeviceMode: £¨0-CPU¡¢1-GPU£©
	=================================================*/
	bool InitModel(const std::wstring &strModelPath, const int nDeviceMode);

	/*================================================
	input:
	@ vImagePath
	output:
	@ vvImageCache: 
	=================================================*/
	bool PreProcess(std::vector<std::vector<float>> &vvImageCache, const std::vector<std::wstring> &vImagePath);

	/*================================================
	input:
	@ vImageCache
	@ pts
	@ vLastMaskCache
	@ OriImgSize: (w,h) order
	output:
	@ vMaskCache: 
	=================================================*/
	bool Execute(std::vector<float> &vMaskCache, const std::vector<float> &vImageCache, const std::vector<LabelPoint> &pts, 
		const std::pair<int, int> &OriImgSize, const std::vector<float> &vLastMaskCache = {});
private:
	AutoLabelingToolImpl* m_pAutoLabelImpl; 
};

#pragma pack(pop)