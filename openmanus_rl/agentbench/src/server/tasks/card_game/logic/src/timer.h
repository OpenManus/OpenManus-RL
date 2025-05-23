#pragma once

#include <chrono>

class Timer {
   public:
    using BaseClock = std::chrono::steady_clock;
    using TimePoint = BaseClock::time_point;
    using Duration = BaseClock::duration;

    Timer() : m_start(Timer::now()) {}
    int runtime() const {
        Duration time = Timer::now() - m_start;
        return std::chrono::duration_cast<std::chrono::seconds>(time).count();
    }
    static TimePoint now() { return BaseClock::now(); }

   private:
    TimePoint m_start;
};