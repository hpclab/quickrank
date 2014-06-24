#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

float rnd01() {
	return rand()/(RAND_MAX+1.0f);
}

#define UNDEFINED_FEATURES_PERCENTAGE 0.0

int main(int argc, char *argv[]) {
	if(argc!=6) {
		printf("usage: %s output_filename #features #samples #min_samples_per_sampleid #max_samples_per_sampleid\n", argv[0]);
		exit(1);
	}
	srand(time(NULL));
	FILE *f = fopen(argv[1], "w");
	if(f) {
		printf("%s : ", argv[1]);
		const int nf = atoi(argv[2]);
		printf("%d features, ", nf);
		const int ns = atoi(argv[3]);
		printf("%d samples, ", ns);
		const int minsxid = atoi(argv[4]);
		const int maxsxid = atoi(argv[5]);
		printf("%d÷%d samples per list... ", minsxid, maxsxid);
		fflush(stdout);
		if(minsxid==0 || nf==0 || ns==0 || maxsxid<minsxid) exit(2);
		int qidlen = minsxid+(rand()%(maxsxid-minsxid+1));
		int qid = 1;
		for(int sid=0; sid!=ns; ++sid, --qidlen) {
			if(qidlen==0)
				qidlen = minsxid+(rand()%(maxsxid-minsxid+1)),
				++qid;
			fprintf(f, "%d.0 qid:%d", rand()%3, qid);
			for(int fid=0; fid!=nf; ++fid)
				if(rnd01()>=UNDEFINED_FEATURES_PERCENTAGE)
					fprintf(f, " %d:%.3f", fid+1, rnd01());
			fprintf(f, " #description of line %d\n", sid);
		}
		fclose(f);
		printf("done\n");
	} else printf("cannot write '%s'\n", argv[1]);
	return 0;
}
