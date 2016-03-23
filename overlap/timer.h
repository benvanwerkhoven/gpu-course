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
 * https://github.com/pathscale/rodinia/blob/master/opencl/bfs/timer.h
 */


#ifndef timer_h
#define timer_h

#include <iostream>

#define createTimer(a) timer a(#a)

class timer {
    public:
			   timer(const char *name = 0);
			   timer(const char *name, std::ostream &write_on_exit);

			   ~timer();

	void		   start(), stop();
	void		   reset();
	std::ostream   	   &print(std::ostream &);

	double             getTimeInSeconds();
	double             getTimeInMilliSeconds();

    private:
	void		   print_time(std::ostream &, const char *which, double time) const;

	union {
	    long long	   total_time;
	    struct {
#if defined __PPC__
		int	   high, low;
#else
		int	   low, high;
#endif
	    };
	};

	unsigned long long count;
	const char	   *const name;
	std::ostream	   *const write_on_exit;

	static double	   CPU_speed_in_MHz, get_CPU_speed_in_MHz();
};


std::ostream &operator << (std::ostream &, class timer &);


inline void timer::reset()
{
    total_time = 0;
    count      = 0;
}


inline timer::timer(const char *name)
:
    name(name),
    write_on_exit(0)
{
    reset();
}


inline timer::timer(const char *name, std::ostream &write_on_exit)
:
    name(name),
    write_on_exit(&write_on_exit)
{
    reset();
}


inline timer::~timer()
{
    if (write_on_exit != 0)
	print(*write_on_exit);
}


inline void timer::start()
{
#if (defined __PATHSCALE__) && (defined __i386 || defined __x86_64)
    unsigned eax, edx;

    asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

    total_time -= ((unsigned long long) edx << 32) + eax;
#elif (defined __GNUC__ || defined __INTEL_COMPILER) && (defined __i386 || defined __x86_64)
    asm volatile
    (
	"rdtsc\n\t"
	"subl %%eax, %0\n\t"
	"sbbl %%edx, %1"
    :
	"+m" (low), "+m" (high)
    :
    :
	"eax", "edx"
    );
#else
#error Compiler/Architecture not recognized
#endif
}


inline void timer::stop()
{
#if (defined __PATHSCALE__) && (defined __i386 || defined __x86_64)
    unsigned eax, edx;

    asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

    total_time += ((unsigned long long) edx << 32) + eax;
#elif (defined __GNUC__ || defined __INTEL_COMPILER) && (defined __i386 || defined __x86_64)
    asm volatile
    (
	"rdtsc\n\t"
	"addl %%eax, %0\n\t"
	"adcl %%edx, %1"
    :
	"+m" (low), "+m" (high)
    :
    :
	"eax", "edx"
    );
#endif

    ++ count;
}

#endif
