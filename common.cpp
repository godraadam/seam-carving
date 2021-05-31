#include "stdafx.h"

#include "common.h"
#include <CommDlg.h>
#include <ShlObj.h>

FileGetter::FileGetter(char* folderin, char* ext){		
	strcpy(folder, folderin);
	char folderstar[MAX_PATH];	
	if( !ext ) strcpy(ext, "*");
	sprintf(folderstar,"%s\\*.%s",folder,ext);
	hfind = FindFirstFileA(folderstar,&found);
	hasFiles= !(hfind == INVALID_HANDLE_VALUE);
	first = 1;
	//skip .
	//FindNextFileA(hfind,&found);		
}

int FileGetter::getNextFile(char* fname){
	if (!hasFiles)
		return 0;
	//skips .. when called for the first time
	if( first )
	{
		strcpy(fname, found.cFileName);				
		first = 0;
		return 1;
	}
	else{
		chk=FindNextFileA(hfind,&found);
		if (chk)
			strcpy(fname, found.cFileName);				
		return chk;
	}
}

int FileGetter::getNextAbsFile(char* fname){
	if (!hasFiles)
		return 0;
	//skips .. when called for the first time
	if( first )
	{
		sprintf(fname, "%s\\%s", folder, found.cFileName);			
		first = 0;
		return 1;
	}
	else{
		chk=FindNextFileA(hfind,&found);
		if (chk)
			sprintf(fname, "%s\\%s", folder, found.cFileName);				
		return chk;
	}
}

char* FileGetter::getFoundFileName(){
	if (!hasFiles)
		return 0;
	return found.cFileName;
}


int openFileDlg(char* fname)
{
	char *filter = "All Files (*.*)\0*.*\0";
	HWND owner = NULL;
	OPENFILENAME ofn;
	char fileName[MAX_PATH];
	strcpy(fileName,"");
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = owner;
	ofn.lpstrFilter = filter;
	ofn.lpstrFile = fileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	ofn.lpstrDefExt = "";
	GetOpenFileName(&ofn); 
	strcpy(fname, ofn.lpstrFile); 
	return strcmp(fname, "");
}

int openFolderDlg(char *folderName)
{
	BROWSEINFO bi;
	ZeroMemory(&bi, sizeof(bi));
	SHGetPathFromIDList(SHBrowseForFolder(&bi),folderName);
	return strcmp(folderName,"");
}

void resizeImg(cv::Mat src, cv::Mat &dst, int maxSize, bool interpolate)
{
	double ratio = 1;
	double w = src.cols;
	double h = src.rows;
	if (w>h)
		ratio = w/(double)maxSize;
	else
		ratio = h/(double)maxSize;
	int nw = (int) (w / ratio);
	int nh = (int) (h / ratio);
	cv::Size sz(nw, nh);
	if (interpolate)
		resize(src, dst, sz);
	else
		resize(src, dst, sz, 0, 0, cv::INTER_NEAREST);
}

void showHistogram(const std::string& name, int* hist, const int hist_cols,
	const int hist_height) {
	cv::Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
	// constructs a white image
   //computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		cv::Point p1 = cv::Point(x, baseline);
		cv::Point p2 = cv::Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
		// colored in magenta
	}
	imshow(name, imgHist);
}