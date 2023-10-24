#!/usr/local/bin/gawk -f
#
# make-diacritic-map --
#	Generate a map from ascii to accented word forms
#	for use with disambig(1)
#
# $Header: /home/srilm/CVS/srilm/utils/src/make-diacritic-map.gawk,v 1.3 1998/02/04 20:28:02 stolcke Exp $
#
/^#/ {
	next;
}
function asciify(word) {
	gsub("�", "A", word);
	gsub("�", "A", word);
	gsub("�", "A", word);
	gsub("�", "A", word);
	gsub("�", "A", word);
	gsub("�", "A", word);
	gsub("�", "AE", word);
	gsub("�", "C", word);
	gsub("�", "E", word);
	gsub("�", "E", word);
	gsub("�", "E", word);
	gsub("�", "E", word);
	gsub("�", "I", word);
	gsub("�", "I", word);
	gsub("�", "I", word);
	gsub("�", "I", word);
	gsub("�", "N", word);
	gsub("�", "O", word);
	gsub("�", "O", word);
	gsub("�", "O", word);
	gsub("�", "O", word);
	gsub("�", "O", word);
	gsub("�", "O", word);
	gsub("�", "U", word);
	gsub("�", "U", word);
	gsub("�", "U", word);
	gsub("�", "U", word);
	gsub("�", "Y", word);
	gsub("�", "ss", word);
	gsub("�", "a", word);
	gsub("�", "a", word);
	gsub("�", "a", word);
	gsub("�", "a", word);
	gsub("�", "a", word);
	gsub("�", "a", word);
	gsub("�", "a", word);
	gsub("�", "c", word);
	gsub("�", "e", word);
	gsub("�", "e", word);
	gsub("�", "e", word);
	gsub("�", "e", word);
	gsub("�", "i", word);
	gsub("�", "i", word);
	gsub("�", "i", word);
	gsub("�", "i", word);
	gsub("�", "n", word);
	gsub("�", "o", word);
	gsub("�", "o", word);
	gsub("�", "o", word);
	gsub("�", "o", word);
	gsub("�", "o", word);
	gsub("�", "u", word);
	gsub("�", "u", word);
	gsub("�", "u", word);
	gsub("�", "u", word);
	gsub("�", "y", word);
	return word;
}
{
	word = $1;
	asciiword = asciify(word);

	if (asciiword in map) {
		map[asciiword] = map[asciiword] " " word;
	} else {
		map[asciiword] = word;
	}
}
END {
	print "<s>\t<s>"
	print "</s>\t</s>"
	fflush()

	for (w in map) {
		print w "\t" map[w] | "sort";
	}
}
