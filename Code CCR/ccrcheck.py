# 通过diff文件计算增加和减少的行数，用于计算CCR impact
# 如何参数可以为单个diff文件名，或包含diff文件的目录
import re
import sys
import os

def analyze(infile):
    addlines = 0
    deletelines = 0
    file = open(infile)
    addpatten = re.compile(r"\+", re.VERBOSE)
    deletepatten = re.compile(r"\-", re.VERBOSE)
    commentpatten = re.compile(r"(\+\*|\+(\s)*//|\-\*|\-(\s)*//)", re.VERBOSE)
    for line in file:
        addMatch = addpatten.match(line)
        if addMatch:
            commentMatch = commentpatten.match(line)
            if not commentMatch:
                addlines = addlines + 1
        else:
            deleteMatch = deletepatten.match(line)
            if deleteMatch:
                commentMatch = commentpatten.match(line)
                if not commentMatch:
                    deletelines = deletelines + 1

    print("\n>>>" + file.name + " analyze result:")
    report(addlines, deletelines)
    return (addlines, deletelines)

def report(addlines, deletelines):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("add lines:", addlines)
    print("-----------------------------------------------------------------")
    print("delete lines:", deletelines)
    print("-----------------------------------------------------------------")
    if addlines > deletelines:
        print("totally add: ", addlines - deletelines, "lines")
    else:
        print("totally delete: ", deletelines - addlines, "lines")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

def startAnalyze(inFile):
    if os.path.isfile(inFile) and inFile.endswith("diff"):
        analyze(inFile)
    elif os.path.isdir(inFile):
        addsum = 0
        deletesum = 0
        for item in os.listdir(inFile):
            if not item.endswith("diff"):
                continue
            tmpa,tmpd = analyze(item)
            addsum = addsum + tmpa
            deletesum = deletesum + tmpd

        print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>analyze fnished, here is the final result for all files:")
        report(addsum, deletesum)

inFile = sys.argv[1]
startAnalyze(inFile)
