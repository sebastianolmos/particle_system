#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

class PerformanceMonitor
{
private:
	float currentTime;
	float timer;
	float period;
	int framesCounter;
	float framesPerSecond;
	float msPerFrame;

public:
	PerformanceMonitor(float ctTime, float prd) :
		currentTime(ctTime),
		timer(0.0f),
		period(prd),
		framesCounter(0),
		framesPerSecond(0.0f),
		msPerFrame(0.0f)
	{}

	void update(float cTime)
	{
		framesCounter += 1;
		timer += cTime - currentTime;
		currentTime = cTime;

		if (timer > period)
		{
			framesPerSecond = framesCounter / timer;
			msPerFrame = 1000.0 * timer / framesCounter;
			framesCounter = 0;
			timer = 0.0f;
		}
	}

	inline float getFPS() const
	{
		return framesPerSecond;
	}

	inline float getMS() const
	{
		return msPerFrame;
	}
};

std::ostream& operator<<(std::ostream& os, const PerformanceMonitor perfMonitor) {
	os << std::fixed << std::setprecision(2)
		<< "[" << perfMonitor.getFPS() << " fps - "
		<< perfMonitor.getMS() << " ms]";
	return os;
}