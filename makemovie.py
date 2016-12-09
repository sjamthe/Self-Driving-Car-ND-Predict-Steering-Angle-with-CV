import os
from moviepy.editor import ImageSequenceClip

def makemovie(dirname, outfile):
	images = []
	filecnt = len(os.listdir(dirname))-1
	for cnt in range(1,filecnt):
		images.append(dirname+str(cnt)+'.jpg')

	clip = ImageSequenceClip(images,fps=30)
	clip.write_videofile(outfile)

makemovie('./input/','in-movie.mp4')
makemovie('./out/','out-movie.mp4')
