#ifndef LOCALSTRUCTSH_INCLUDED
#define LOCALSTRUCTSH_INCLUDED

#ifdef SLOW
#include "defaultstructs.h"
#endif
#ifdef FAST
#include "defaultstructs.h"
#endif
#ifdef ALTIVEC
#include "altivecstructs.h"
#endif

#endif /*LOCALSTRUCTSH_INCLUDED*/