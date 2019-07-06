import os
import sys
import volatility.utils as utils
import volatility.plugins.common as common
import volatility.plugins.taskmods as taskmods
from time import gmtime, strftime

PAGES_OUTPUT_PATH = '../create_memory_graphs/pages/'
WIN32_OR_64 = 32

class get_physical_pages(taskmods.DllList):
    '''Get page talbe mapping information'''

    def __init__(self, config, *args, **kwargs):
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        self.kernel_address_space = None
        config.add_option("VIRTUAL", short_option = "V", default = True, action = "store_true", help = "Scan virtual space instead of physical")

    def get_continuous_pages(self, available_pages):
        dict_page_addr_to_size = {}
        dict_tail_to_page_head = {}
        for (page_addr, page_size) in available_pages:
            if WIN32_OR_64 == 64:
                page_addr += 0xffff000000000000
                kernel_space_start_addr = 0xffff080000000000
            else:
                kernel_space_start_addr = 0x80000000
            if page_addr > kernel_space_start_addr:
                if page_addr not in dict_tail_to_page_head:
                    dict_tail_to_page_head[page_addr + page_size] = page_addr
                    dict_page_addr_to_size[page_addr] = page_size
                else:
                    page_head = dict_tail_to_page_head[page_addr]
                    dict_tail_to_page_head[page_addr + page_size] = page_head
                    dict_page_addr_to_size[page_head] += page_size
        return dict_page_addr_to_size

    def calculate(self):
        image_name = os.path.basename(self._config.LOCATION)
        self.log(image_name)

        self.kernel_address_space = utils.load_as(self._config)
        available_pages = self.kernel_address_space.get_available_pages()

        #self.log('Get continuous pages')
        #dict_page_addr_to_size = self.get_continuous_pages(available_pages)
        dict_page_addr_to_size = {}
        for addr, size in available_pages:
            if addr > 0x80000000:
                dict_page_addr_to_size[addr] = size

        with open(PAGES_OUTPUT_PATH + 'pages.' + image_name, 'w') as output:
            list_addr = dict_page_addr_to_size.keys()
            list_addr.sort()
            for addr in list_addr:
                size = dict_page_addr_to_size[addr]
                physical_addr = self.kernel_address_space.vtop(addr)
                output.write(str(addr) + '\t' + str(physical_addr) + '\t' + str(size) + '\n')
        self.log('Finish')

    def render_text(self, outfd, data):
        if data!=None:
            outfd.write(data)

    def log(self, message):
        print strftime("%Y-%m-%d %H:%M:%S\t", gmtime()), message
        sys.stdout.flush()
