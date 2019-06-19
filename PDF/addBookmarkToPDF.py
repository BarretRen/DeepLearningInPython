# -*- coding: utf-8 -*- 
# 封装的PDF文档书签生成工具，需要提供PDF文件和书签TXT文件作为输入
# 目前支持三级目录分级
from PyPDF2 import PdfFileReader as reader,PdfFileWriter as writer
import os

class PDFHandleMode(object):
    '''
    处理PDF文件的模式
    '''
    # 保留源PDF文件的所有内容和信息，在此基础上修改
    COPY = 'copy'
    # 仅保留源PDF文件的页面内容，在此基础上修改
    NEWLY = 'newly'

class MyPDFHandler(object):
    '''
    封装的PDF文件处理类
    '''
    def __init__(self,pdf_file_path,mode = PDFHandleMode.COPY):
        '''
        用一个PDF文件初始化
        :param pdf_file_path: PDF文件路径
        :param mode: 处理PDF文件的模式，默认为PDFHandleMode.COPY模式
        '''
        # 只读的PDF对象
        self.__pdf = reader(pdf_file_path)

        # 获取PDF文件名（不带路径）
        self.file_name = os.path.basename(pdf_file_path)
        self.metadata = self.__pdf.getXmpMetadata()
        self.doc_info = self.__pdf.getDocumentInfo()
        self.pages_num = self.__pdf.getNumPages()

        # 可写的PDF对象，根据不同的模式进行初始化
        self.__writeable_pdf = writer()
        if mode == PDFHandleMode.COPY:
            self.__writeable_pdf.cloneDocumentFromReader(self.__pdf)
        elif mode == PDFHandleMode.NEWLY:
            for idx in range(self.pages_num):
                page = self.__pdf.getPage(idx)
                self.__writeable_pdf.insertPage(page, idx)

    def save2file(self,new_file_name):
        '''
        将修改后的PDF保存成文件
        :param new_file_name: 新文件名，不要和原文件名相同
        :return: None
        '''
        # 保存修改后的PDF文件内容到文件中
        with open(new_file_name, 'wb') as fout:
            self.__writeable_pdf.write(fout)
        print('save2file success! new file is: {0}'.format(new_file_name))

    def add_one_bookmark(self,title,page,parent = None, color = None,fit = '/Fit'):
        '''
        往PDF文件中添加单条书签，并且保存为一个新的PDF文件
        :param str title: 书签标题
        :param int page: 书签跳转到的页码，表示的是PDF中的绝对页码，值为1表示第一页
        :paran parent: A reference to a parent bookmark to create nested bookmarks.
        :param tuple color: Color of the bookmark as a red, green, blue tuple from 0.0 to 1.0
        :param list bookmarks: 是一个'(书签标题，页码)'二元组列表，举例：[(u'tag1',1),(u'tag2',5)]，页码为1代表第一页
        :param str fit: 跳转到书签页后的缩放方式
        :return: None
        '''
        bookmark = self.__writeable_pdf.addBookmark(title,page - 1,parent = parent,color = color,fit = fit)
        print('add_one_bookmark success! bookmark title is: {0}'.format(title))
        return bookmark

    def add_bookmarks(self,bookmarks):
        '''
        批量添加书签
        :param bookmarks: 书签元组列表，其中的页码表示的是PDF中的绝对页码，值为1表示第一页
        :return: None
        '''
        l1 = None
        l2 = None
        for level,title,page in bookmarks:
            if (level == 1):
                l1 = self.add_one_bookmark(title,page)
            elif (level == 2):
                l2 = self.add_one_bookmark(title,page, l1)
            else:
                self.add_one_bookmark(title,page, l2)
        print('add_bookmarks success! add {0} pieces of bookmarks to PDF file'.format(len(bookmarks)))

    def read_bookmarks_from_txt(self,txt_file_path,page_offset = 0):
        '''
        从文本文件中读取书签列表
        文本文件有若干行，每行一个书签，内容格式为：
        书签标题 页码
        注：中间用空格隔开，页码为1表示第1页
        :param txt_file_path: 书签信息文本文件路径
        :param page_offset: 页码便宜量，为0或正数，即由于封面、目录等页面的存在，在PDF中实际的绝对页码比在目录中写的页码多出的差值
        :return: 书签列表
        '''
        bookmarks = []
        with open(txt_file_path,'r',encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip()
                if not line:
                    continue
                # 以'@'作为标题、页码分隔符,以' '作为章节号和内容的分隔符
                print('read line is: {0}'.format(line))
                try:
                    chapter = line.split(' ')[0].rstrip()
                    level = len(chapter.split('.'))
                    
                    title = line.split('@')[0].rstrip()
                    page = line.split('@')[1].strip()
                except IndexError as msg:
                    print(msg)
                    continue
                # level,title和page都不为空才添加书签，否则不添加
                if level and title and page:
                    try:
                        page = int(page) + page_offset
                        bookmarks.append((level, title, page))
                    except ValueError as msg:
                        print(msg)

        return bookmarks

    def add_bookmarks_by_read_txt(self,txt_file_path,page_offset = 0):
        '''
        通过读取书签列表信息文本文件，将书签批量添加到PDF文件中
        :param txt_file_path: 书签列表信息文本文件
        :param page_offset: 页码便宜量，为0或正数，即由于封面、目录等页面的存在，在PDF中实际的绝对页码比在目录中写的页码多出的差值
        :return: None
        '''
        bookmarks = self.read_bookmarks_from_txt(txt_file_path,page_offset)
        self.add_bookmarks(bookmarks)
        print('add_bookmarks_by_read_txt success!')

def main():
    pdf_handler = MyPDFHandler(u'C:/N-20L6PF1PYK28-Data/barretr/Desktop/TCPIP详解卷一.pdf',PDFHandleMode.NEWLY)
    pdf_handler.add_bookmarks_by_read_txt('C:/N-20L6PF1PYK28-Data/barretr/Desktop/1.txt')
    pdf_handler.save2file(u'C:/N-20L6PF1PYK28-Data/barretr/Desktop/TCPIP详解卷一-目录书签版.pdf')

if __name__ == '__main__':
    main()
