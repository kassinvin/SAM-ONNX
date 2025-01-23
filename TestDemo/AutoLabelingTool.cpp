#include "AutoLabelingTool.h"
#include "AutoLabelingToolImpl.h"


AutoLabelingTool::AutoLabelingTool()
{
	m_pAutoLabelImpl = new(std::nothrow) AutoLabelingToolImpl;
}

AutoLabelingTool::~AutoLabelingTool()
{
	if (m_pAutoLabelImpl != nullptr)
	{
		delete m_pAutoLabelImpl;
		m_pAutoLabelImpl = nullptr;
	}
}

bool AutoLabelingTool::InitModel(const std::wstring &strModelPath, const int nDeviceMode)
{
	return m_pAutoLabelImpl ? m_pAutoLabelImpl->InitModel(strModelPath, nDeviceMode) : false;
}

bool AutoLabelingTool::PreProcess(std::vector<std::vector<float>> &vvImageCache, const std::vector<std::wstring> &vImagePath)
{
	return m_pAutoLabelImpl ? m_pAutoLabelImpl->PreProcess(vvImageCache, vImagePath) : false;
}

bool AutoLabelingTool::Execute(std::vector<float> &vMaskCache, 
	const std::vector<float> &vImageCache, const std::vector<LabelPoint> &pts, const std::pair<int, int> &OriImgSize,
	const std::vector<float> &vLastMaskCache /*= {}*/)
{
	return m_pAutoLabelImpl ? m_pAutoLabelImpl->Execute(vMaskCache, vImageCache, pts, OriImgSize, vLastMaskCache) : false;
}