# 从repo中获取所有相关的修改集(指定keyword,一般是项目的IR号），并计算最终的代码添加量
# 参数为-r/--repo,-i/--ir,-b/--branch 
from mercurial import util, commands
from mercurial import ui, hg
import os
import sys
import re
import argparse

def parseArg():
    argParser = argparse.ArgumentParser(description='get diff files base on the IR number',
            epilog='example: ./irDiffGet.py -r /repo/barretr/fdt1265/sw -i ALU02376346 -b default')

    argParser.add_argument('-r', '--repo', default=None, help='specify the repo path')
    argParser.add_argument('-i', '--ir', default=None, help='specify the IR number')
    argParser.add_argument('-b', '--branch', default='default', help='specify the branch')

    args = argParser.parse_args()

    if args.repo is None and args.ir is None:
        argParser.error('No repo and ir number specified, add -r/--repo or -i/--ir')

    return args

def getCodeModify(args):
    repoui=ui.ui()
    repo=hg.repository(repoui, args.repo)
    ir=args.ir
    branch=args.branch
    # get all changeset for the IR and branch
    repoui.pushbuffer()
    commands.log(repoui, repo, template='{node};{desc}\n', keyword=[ir], branch=[branch])
    nodeList=repoui.popbuffer().split('\n')

    mergeMatch=re.compile(r'(rebase|merge\s)', re.IGNORECASE)
    addMatch=re.compile(r'\+([0-9]\d*)/\-([0-9]\d*)')
    codeLines=0
    for node in nodeList:
        result=mergeMatch.findall(node)
        if result:
            continue
        
        changeset=node.split(';')[0]
        if changeset == "":
            continue

        repoui.pushbuffer()
        commands.log(repoui, repo, template='{diffstat}\n', rev=[changeset])
        diff=repoui.popbuffer().split('\n') # always one element in the list
        lines=addMatch.search(diff[0])
        if lines:
            print(changeset,"add lines:", lines.group(1),"delete lines", lines.group(2))
            codeLines = codeLines+int(lines.group(1))-int(lines.group(2))

    print("totally add lines", codeLines)


def main():
    args = parseArg()
    getCodeModify(args)

main()
