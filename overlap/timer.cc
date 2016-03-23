/*
 * LICENSE TERMS
 *
 * Copyright (c)2008-2011 University of Virginia
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted without royalty 
 * fees or other restrictions, provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this list of conditions and the 
 *     following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
 *     following disclaimer in the documentation and/or other materials provided with the distribution.
 * * Neither the name of the University of Virginia, the Dept. of Computer Science, nor the names of its 
 *     contributors may be used to endorse or promote products derived from this software without specific prior
 *     written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF VIRGINIA OR THE SOFTWARE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * https://github.com/pathscale/rodinia/blob/master/opencl/bfs/timer.cc
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>


using namespace std;

#include "timer.h"



double timer::CPU_speed_in_MHz = timer::get_CPU_speed_in_MHz();


double timer::get_CPU_speed_in_MHz()
{
#if defined __linux__
    ifstream infile("/proc/cpuinfo");
    char     buffer[256], *colon;

    while (infile.good()) {
	infile.getline(buffer, 256);

	if (strncmp("cpu MHz", buffer, 7) == 0 && (colon = strchr(buffer, ':')) != 0)
	    return atof(colon + 2);
    }
#endif

    return 0.0;
}


void timer::print_time(ostream &str, const char *which, double time) const
{
    static const char *units[] = { " ns", " us", " ms", "  s", " ks", 0 };
    const char	      **unit   = units;

    time = 1000.0 * time / CPU_speed_in_MHz;

    while (time >= 999.5 && unit[1] != 0) {
	time /= 1000.0;
	++ unit;
    }

    str << which << " = " << setprecision(3) << setw(4) << time << *unit;
}


ostream &timer::print(ostream &str)
{
    str << left << setw(25) << (name != 0 ? name : "timer") << ": " << right;

    if (CPU_speed_in_MHz == 0)
	str << "could not determine CPU speed\n";
    else if (count > 0) {
	double total = static_cast<double>(total_time);

	print_time(str, "avg", total / static_cast<double>(count));
	print_time(str, ", total", total);
	str << ", count = " << setw(9) << count << '\n';
    }
    else
	str << "not used\n";

    return str;
}


ostream &operator << (ostream &str, class timer &timer)
{
    return timer.print(str);
}

double timer::getTimeInSeconds()
{
    double total = static_cast<double>(total_time);
    double res = (total / 1000000.0) / CPU_speed_in_MHz;
    return res;
}

double timer::getTimeInMilliSeconds()
{
    double total = static_cast<double>(total_time);
    double res = (total / 1000.0) / CPU_speed_in_MHz;
    return res;
}



/*
 * The following was added to enable easy use of this handy timer outside of C++
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */

createTimer(timer);

extern "C" {

void start_timer() {
  timer.reset();
  timer.start();
}

void stop_timer(float *time) {
  timer.stop();
  *time = (float)timer.getTimeInMilliSeconds();
}


}


