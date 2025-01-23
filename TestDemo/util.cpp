#include "util.h"
#include <Windows.h>

std::string Ws2S(const std::wstring& s)
{
    const DWORD dwNum = WideCharToMultiByte(CP_OEMCP, NULL, s.c_str(), -1, nullptr, 0, nullptr, nullptr);
    char* buf;
    buf = new char[dwNum];
    if (buf == nullptr)
    {
        delete[] buf;
        buf = nullptr;
    }
    WideCharToMultiByte(CP_OEMCP, NULL, s.c_str(), -1, buf, dwNum, nullptr, nullptr);
    //std::string r = buf;

    std::string r;
    r.resize(dwNum);
    memcpy((void*)r.data(), buf, sizeof(char) * dwNum);

    delete[] buf;
    buf = nullptr;

    r = r.substr(0, r.length() - 1);

    return std::move(r);
}