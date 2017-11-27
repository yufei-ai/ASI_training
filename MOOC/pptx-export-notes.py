#!/usr/bin/env python
import os, glob
from zipfile import ZipFile
from xml.dom.minidom import parse
import shutil

def slide_number_from_xml_file(filename):
    """
    Integer slide number from filename
    Assumes /path/to/Slidefile/somekindofSlide36.something
    """
    return int(filename[filename.rfind("Slide") + 5:filename.rfind(".")])

def getSlideNumber(x):
    a = x.split('\\')
    f = a[0]+'/_rels/'+a[1]+'.rels'
    dom = parse(f)
    a=dom.getElementsByTagName('Relationships')[0]
    b=a.firstChild
    return int(b.getAttribute('Target')[15:-4])


#main function
def run(fname):
    notesDict = {}
    path = '/tmp/ppt/notesSlides/'
    shutil.rmtree(path, ignore_errors=True)
    ZipFile(fname).extractall(path='/tmp/', pwd=None)

    #open up the file that you wish to write to
    writepath = os.path.dirname(fname) + '/' + os.path.basename(fname).rsplit('.', 1)[
        0] + '_notes.txt'
    # Get the xml we extracted from the zip file
    xmlfiles = glob.glob(os.path.join(path, '*.xml'))
    with open(writepath, 'w') as f:
        for infile in sorted(xmlfiles, key=slide_number_from_xml_file):
            #parse each XML notes file from the notes folder.
            dom = parse(infile)
            noteslist = dom.getElementsByTagName('a:t')
            if len(noteslist) == 0:
                continue

            #separate last element of noteslist for use as the slide marking.
            slideNumber = getSlideNumber(infile)
            #start with this empty string to build the presenter note itself
            tempstring = ''

            for node in noteslist:
                xmlTag = node.toxml()
                xmlData = xmlTag.replace('<a:t>', '').replace('</a:t>', '')
                #concatenate the xmlData to the tempstring for the particular slideNumber index.
                tempstring = tempstring + xmlData

            #store the tempstring in the dictionary under the slide number
            notesDict[slideNumber] = tempstring

        #print/write the dictionary to file in sorted order by key value.
        for x in [key for key in sorted(notesDict.keys(), key=int)]:
            f.write('Slide ' + str(x) + '\n')
            notes_string = notesDict[x]
            f.write(notes_string.encode('utf-8', 'ignore') + '\n')
#            print notes_string 
        print 'file successfully written to' + '\'' + writepath + '\''

fname = 'C:/consulting/asi/mooc/Chapter2New.pptx'
run(fname)