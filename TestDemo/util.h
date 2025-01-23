#pragma once
#include <string>
#include <vector>
#include <time.h>

struct LabelPoint
{
    int m_nType;//£¨0-negative£¬1-positive£©
    int m_nX;	//x
    int m_nY;    //y

    LabelPoint() : m_nType(-1), m_nX(-1), m_nY(-1)
    {

    };

    LabelPoint(int nType, int nX, int nY)
    {
        m_nType = nType;
        m_nX = nX;
        m_nY = nY;
    };
};

template <unsigned int N>
const char* Timestamp(char(&buf)[N])
{
    time_t timep;
    time(&timep);
    strftime(buf, _countof(buf), "%m/%d/%Y %H:%M:%S", localtime(&timep));
    return buf;
}

#define PREPENDTS(stream) \
    do \
    { \
        if (true) \
        { \
            char mbstr[30]; \
            fprintf(stream, "%s: ", Timestamp(mbstr));  \
        } \
    } while(0)

#define LOGPRINTF(stream, ...) \
    do \
    { \
        PREPENDTS(stream); \
        fprintf(stream, __VA_ARGS__); \
    } while(0)


// wstring to string
std::string Ws2S(const std::wstring& s);