import os
import volatility.utils as utils
import volatility.plugins.taskmods as taskmods
OUTPUT_PATH = '../create_memory_graphs/label/'

dict_object_to_structure = {'File':'_FILE_OBJECT', \
                            'Process':'_EPROCESS', \
                            'Thread':'_ETHREAD', \
                            'Driver':'_DRIVER_OBJECT', \
                            'Key':'_CM_KEY_BODY'}

class my_handles(taskmods.DllList):
    '''Grabeach address in the given memory dump'''

    def __init__(self, config, *args, **kwargs):
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        self.kernel_address_space = None
        config.add_option("VIRTUAL", short_option = "V", default = True, action = "store_true", help = "Scan virtual space instead of physical")

    def handle_scan(self):
        dict_addr_to_type = {}
        dict_type_to_size = {}
        for task in taskmods.DllList.calculate(self):
            if task.ObjectTable.HandleTableList:
                for handle in task.ObjectTable.handles():
                    if not handle.is_valid():
                        continue
                    object_size = 0
                    object_type = handle.get_object_type()
                    #object_offset = handle.obj_vm.vtop(handle.Body.obj_offset)
                    object_offset = handle.Body.obj_offset
                    if object_type in dict_object_to_structure:
                        object_size = handle.obj_vm.profile.get_obj_size(dict_object_to_structure[object_type])
                        dict_type_to_size[object_type] = object_size
                        dict_addr_to_type[object_offset] = object_type
        return dict_addr_to_type, dict_type_to_size

    def calculate(self):
        image_name = os.path.basename(self._config.LOCATION)
        self.kernel_address_space = utils.load_as(self._config)

        dict_addr_to_type, dict_type_to_size = self.handle_scan()
        addrs = dict_addr_to_type.keys()
        addrs.sort()
        with open(OUTPUT_PATH + 'label.' + image_name + '.handles', 'w') as f:
            for addr in addrs:
                f.write(hex(addr) + '\t' + dict_object_to_structure[dict_addr_to_type[addr]] + '\t' + str(dict_type_to_size[dict_addr_to_type[addr]]) + '\n')

    def render_text(self, outfd, data):
        return
