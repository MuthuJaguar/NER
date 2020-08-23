folder<-file.path("F:\\Machine_Learning\\papers")
folder
length<-length(dir(folder))
length
dirpdf<-dir(folder)
dirpdf[1]

pdftotxt<-"C:\\Program Files\\xpdf-tools-win-4.02\\bin64\\pdftotext.exe"

for(i in 1:length(dir(folder)))
{
  pdf<-file.path("F:\\Machine_Learning\\papers",dirpdf[i])
  system(paste("\"", pdftotxt, "\" \"", pdf,"\"", sep =""), wait=FALSE)
}
