import os
import volatility.utils as utils
import volatility.plugins.common as common
import volatility.plugins.taskmods as taskmods
from time import gmtime, strftime
import volatility.plugins.my_handles

ROOT_PATH = '/rhome/wsong008/bigdata/kernel_object_recognition/'
LABEL_PATH = ROOT_PATH + 'build_graph/label/'

class print_image(taskmods.DllList):
    '''Grabeach address in the given memory dump'''

    def __init__(self, config, *args, **kwargs):
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        self.kernel_address_space = None
        config.add_option("VIRTUAL", short_option = "V", default = True, action = "store_true", help = "Scan virtual space instead of physical")

    def calculate(self):
        image_name = os.path.basename(self._config.LOCATION)
        self.log(image_name)
        self.kernel_address_space = utils.load_as(self._config)

        dict_vaddr_to_label = self.get_vaddr_to_label(image_name)

        available_pages = self.kernel_address_space.get_available_pages()
        self.print_image_content(available_pages, image_name, dict_vaddr_to_label)

    def content_to_hex(self, content):
        hexs = []
        for char in content:
            hexs.append(str(hex(ord(char))).replace('0x', '').zfill(2))
        return hexs

    def to_hex(self, content):
        hexs = self.content_to_hex(content)
        out = ''
        for index in range(len(hexs)):
            out += hexs[index] + ' ' 
        return out.strip()

    def to_ascii(self, content):
        out = ''
        for index in range(len(content)):
            num = ord(content[index])
            if num >= 32 and num <= 127:
                out += str(unichr(num))
            else:
                out += ' '
        return out

    def render_text(self, outfd, data):
        if data!=None:
            outfd.write(data)

    def print_image_content(self, available_pages, image_name, dict_vaddr_to_label):
        dict_ptr_to_dest = {}
        set_page_break = set()
        list_process = []
        for vaddr, label in dict_vaddr_to_label.iteritems():
            if label == '_EPROCESS':
                list_process.append(vaddr)
        with open('/rhome/wsong008/bigdata/kernel_object_recognition/mutate/image_content/content.' + image_name, 'w') as f:
            for (page_addr, size) in available_pages:
                #page_addr = page_addr + 0xffff000000000000
                #if page_addr > 0xffff080000000000:
                if page_addr > 0x80000000:
                    buf = self.kernel_address_space.read(page_addr, size)
                    if buf != None:
                        for idx in range(0, size, 4):
                            f.write(hex(page_addr + idx) + '\t' + self.to_hex(buf[idx:idx+4]) + '\t' + self.to_ascii(buf[idx:idx+4]))
                            if page_addr+idx in dict_vaddr_to_label:
                                f.write('\t' + dict_vaddr_to_label[page_addr+idx])
                            elif ord(buf[idx+3]) > 0x80 and ord(buf[idx]) % 4 == 0:
                                dest = (ord(buf[idx+3]) << 24) + (ord(buf[idx+2]) << 16) + (ord(buf[idx+1]) << 8) + ord(buf[idx])
                                #if dest in dict_vaddr_to_label:
                                #    f.write('\t--> ' + dict_vaddr_to_label[dest])
                                for vaddr in list_process:
                                    if dest > vaddr and dest - vaddr < 704:
                                        f.write('\t--> active process ' + str(hex(vaddr)))
                            f.write('\n')

    def get_vaddr_to_label(self, image_name):
        image_name = image_name.split('.mutate')[0]
        dict_vaddr_to_label = {}
        set_vaddr_type_size = set()
        self.read_label_file(image_name, set_vaddr_type_size, 'multiscan')
        self.read_label_file(image_name, set_vaddr_type_size, 'handles')
        print len(set_vaddr_type_size)
        for vaddr, obj_type, obj_size in set_vaddr_type_size:
            dict_vaddr_to_label[vaddr] = obj_type
        return dict_vaddr_to_label

    def read_label_file(self, image_name, set_vaddr_type_size, type):
        image_name = image_name.split('.mutate')[0]
        with open(LABEL_PATH + 'label.' + image_name + '.' + type, 'r') as f:
            for line in f:
                s = line.strip().split('\t')
                vaddr = int(s[0], 0)
                obj_type = s[1]
                obj_size = int(s[2])
                record = (vaddr, obj_type, obj_size)
                if record not in set_vaddr_type_size:
                    set_vaddr_type_size.add(record)

    def log(self, message):
        print strftime("%Y-%m-%d %H:%M:%S\t", gmtime()), message
