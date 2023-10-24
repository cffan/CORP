/*
 * rand48.c --
 *	Replacement *rand48 functions (for systems that don't have them)
 *
 * $Header: /home/srilm/CVS/srilm/misc/src/rand48.c,v 1.3 2016/05/25 16:26:15 stolcke Exp $
 */

#ifdef NEED_RAND48

/************************************************************************
 *                                                                      *
 * Copyright (c) 1993 Martin Birgmeier                                  *
 * All rights reserved.                                                 *
 *                                                                      *
 * You may redistribute unmodified or modified versions of this source  *
 * code provided that the above copyright notice and this and the       *
 * following conditions are retained.                                   *
 *                                                                      *
 * This software is provided ``as is'', and comes with no warranties    *
 * of any kind. I shall in no event be liable for anything that happens *
 * to anyone/anything when using this software.                         *
 *                                                                      *
 ************************************************************************/

#include <math.h>
#include <stdlib.h>

#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

unsigned short _rand48_seed[3] = {
		RAND48_SEED_0,
		RAND48_SEED_1,
		RAND48_SEED_2
};
unsigned short _rand48_mult[3] = {
		RAND48_MULT_0,
		RAND48_MULT_1,
		RAND48_MULT_2
};
unsigned short _rand48_add = RAND48_ADD;

void
_dorand48(unsigned short xseed[3])
{
		unsigned long accu;
		unsigned short temp[2];

		accu = (unsigned long) _rand48_mult[0] * (unsigned long) xseed[0] +
		 (unsigned long) _rand48_add;
		temp[0] = (unsigned short) accu;        /* lower 16 bits */
		accu >>= sizeof(unsigned short) * 8;
		accu += (unsigned long) _rand48_mult[0] * (unsigned long) xseed[1] +
		 (unsigned long) _rand48_mult[1] * (unsigned long) xseed[0];
		temp[1] = (unsigned short) accu;        /* middle 16 bits */
		accu >>= sizeof(unsigned short) * 8;
		accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
		xseed[0] = temp[0];
		xseed[1] = temp[1];
		xseed[2] = (unsigned short) accu;
}

double
erand48(unsigned short xseed[3])
{
		_dorand48(xseed);
		return ldexp((double) xseed[0], -48) +
			   ldexp((double) xseed[1], -32) +
			   ldexp((double) xseed[2], -16);
}

double
drand48(void)
{
		return erand48(_rand48_seed);
}

long
lrand48(void)
{
		_dorand48(_rand48_seed);
		return ((long) _rand48_seed[2] << 15) + ((long) _rand48_seed[1] >> 1);
}

long
nrand48(unsigned short xseed[3])
{
		_dorand48(xseed);
		return ((long) xseed[2] << 15) + ((long) xseed[1] >> 1);
}

long
mrand48(void)
{
		_dorand48(_rand48_seed);
		return ((long) _rand48_seed[2] << 16) + (long) _rand48_seed[1];
}

long
jrand48(unsigned short xseed[3])
{
		_dorand48(xseed);
		return ((long) xseed[2] << 16) + (long) xseed[1];
}

void
srand48(long seed)
{
		_rand48_seed[0] = RAND48_SEED_0;
		_rand48_seed[1] = (unsigned short) seed;
		_rand48_seed[2] = (unsigned short) (seed >> 16);
		_rand48_mult[0] = RAND48_MULT_0;
		_rand48_mult[1] = RAND48_MULT_1;
		_rand48_mult[2] = RAND48_MULT_2;
		_rand48_add = RAND48_ADD;
}

unsigned short *
seed48(unsigned short xseed[3])
{
		static unsigned short sseed[3];

		sseed[0] = _rand48_seed[0];
		sseed[1] = _rand48_seed[1];
		sseed[2] = _rand48_seed[2];
		_rand48_seed[0] = xseed[0];
		_rand48_seed[1] = xseed[1];
		_rand48_seed[2] = xseed[2];
		_rand48_mult[0] = RAND48_MULT_0;
		_rand48_mult[1] = RAND48_MULT_1;
		_rand48_mult[2] = RAND48_MULT_2;
		_rand48_add = RAND48_ADD;
		return sseed;
}

void
lcong48(unsigned short p[7])
{
		_rand48_seed[0] = p[0];
		_rand48_seed[1] = p[1];
		_rand48_seed[2] = p[2];
		_rand48_mult[0] = p[3];
		_rand48_mult[1] = p[4];
		_rand48_mult[2] = p[5];
		_rand48_add = p[6];
}

#endif /* NEED_RAND48 */
