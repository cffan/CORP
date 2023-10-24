/*
 * NonzeroLM.h --
 *	Wrapper language model to ensure nonzero probabilities
 *
 * The LM looks up a different word if a given word gives a zero probability.
 * Also, it allows overriding the probability of the unknown word.
 *
 * Copyright (c) 2011 SRI International, 2017 Andreas Stolcke, Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/NonzeroLM.h,v 1.4 2019/09/09 23:13:13 stolcke Exp $
 *
 */

#ifndef _NonzeroLM_h_
#define _NonzeroLM_h_

#include "LM.h"

class NonzeroLM: public LM
{
public:
    NonzeroLM(Vocab &vocab, LM &lm, VocabString zerowordString, LogP unkProb = LogP_Zero);

    /*
     * LM interface
     */
    virtual LogP wordProb(VocabIndex word, const VocabIndex *context);
    virtual void *contextID(VocabIndex word, const VocabIndex *context,
							unsigned &length);
    virtual LogP contextBOW(const VocabIndex *context, unsigned length);

    virtual Boolean isNonWord(VocabIndex word);
    virtual void setState(const char *state);

    virtual Boolean addUnkWords();

    /*
     * Propagate changes to running state to wrapped models
     */
    virtual Boolean running() const { return _running; }
    virtual Boolean running(Boolean newstate)
      { Boolean old = _running; _running = newstate; 
	lm.running(newstate); return old; };

    /*
     * Propagate changes to Debug state to wrapped models
     */
    void debugme(unsigned level)
	{ lm.debugme(level); Debug::debugme(level); };
    ostream &dout() const { return Debug::dout(); };
    ostream &dout(ostream &stream)
	{ lm.dout(stream); return Debug::dout(stream); };

protected:
    LM &lm;				/* wrapped model */
    VocabIndex zeroword;		/* word to back off to */
    LogP unkProb;			/* log prob for <unk> */
};


#endif /* _NonzeroLM_h_ */
