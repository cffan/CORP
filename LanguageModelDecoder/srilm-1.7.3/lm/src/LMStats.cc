/*
 * LMStats.cc --
 *	Generic methods for LM statistics
 *
 */

#ifndef lint
static char LMStats_Copyright[] = "Copyright (c) 1995-2012 SRI International, 2008-2017 Andreas Stolcke, Microsoft Corp.  All Rights Reserved.";
static char LMStats_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/LMStats.cc,v 1.19 2019/09/09 23:13:13 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <string.h>

#include "File.h"
#include "LMStats.h"

#include "LHash.cc"
#include "Trie.cc"
#include "TLSWrapper.h"

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(VocabIndex, unsigned int);
#endif

/*
 * Debug levels used
 */

#define DEBUG_PRINT_TEXTSTATS	1

LMStats::LMStats(Vocab &vocab)
    : vocab(vocab), openVocab(true)
{
    addSentStart = true;
    addSentEnd = true;
}

LMStats::~LMStats()
{
}

static TLSW_ARRAY(VocabString, countStringWords, maxWordsPerLine + 1);
// parse strings into words and update stats
// weighted == 1 indicates each line begins with a count weight
// weighted == 2 indicates each line end with a count weight
unsigned int
LMStats::countString(char *sentence, unsigned weighted)
{
    unsigned int howmany;
    VocabString *words = TLSW_GET_ARRAY(countStringWords);
    
    howmany = vocab.parseWords(sentence, words, maxWordsPerLine + 1);

    if (howmany == maxWordsPerLine + 1) {
	return 0;
    } else {
	if (weighted == 1) {
	    return countSentence(words + 1, words[0]);
	} else if (weighted >= 2) {
	    VocabString weightString = words[howmany - 1];
	    words[howmany - 1] = 0;
	    return countSentence(words, weightString);
	} else {
	    return countSentence(words);
	}
    }
}

void
LMStats::freeThread() 
{
    TLSW_FREE(countStringWords);
}

// parse file into sentences and update stats
unsigned int
LMStats::countFile(File &file, unsigned weighted)
{
    unsigned numWords = 0;
    char *line;

    while ((line = file.getline())) {
	unsigned int howmany = countString(line, weighted);

	/*
	 * Since getline() returns only non-empty lines,
	 * a return value of 0 indicates some sort of problem.
	 */
	if (howmany == 0) {
	    file.position() << (weighted ? "illegal count weight or " : "")
			    << "line too long?\n";
	} else {
	    numWords += howmany;
	}
    }
    if (debug(DEBUG_PRINT_TEXTSTATS)) {
	file.position(dout()) << this -> stats;
    }
    return numWords;
}

