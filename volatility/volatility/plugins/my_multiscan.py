import os
import volatility.utils as utils
import volatility.plugins.common as common
from time import gmtime, strftime
import volatility.win32.tasks as tasks
#import volatility.plugins.malware.threads as threads

import volatility.plugins.filescan as filescan
import volatility.plugins.modscan as modscan
import volatility.plugins.gui.atoms as atoms
import volatility.plugins.gui.windowstations as windowstations
import volatility.plugins.sockscan as sockscan
import volatility.plugins.connscan as connscan
import volatility.plugins.netscan as netscan
import volatility.plugins.malware.callbacks as callbacks
import volatility.plugins.taskmods as taskmods
from volatility.renderers.basic import Address

OUTPUT_PATH = '../create_memory_graphs/label/'
VIRTUAL = True

class my_multiscan(common.AbstractScanCommand):
    '''Grabeach address in the given memory dump'''

    def __init__(self, config, *args, **kwargs):
        common.AbstractScanCommand.__init__(self, config, *args, **kwargs)
        self.kernel_address_space = None
        config.add_option("VIRTUAL", short_option = "V", default = VIRTUAL, action = "store_true", help = "Scan virtual space instead of physical")

    def process_scan(self):
        #addr_space = self.kernel_address_space
        addr_space = utils.load_as(self._config)
        all_tasks = list(tasks.pslist(addr_space))
        dict_process_addr_to_size = dict((p.obj_offset, p.size()) for p in all_tasks)
        list_process_addr = dict_process_addr_to_size.keys()
        return list_process_addr, dict_process_addr_to_size[list_process_addr[0]]

    def multiscan(self):
        dict_addr_to_type ={}
        dict_type_to_size = {}
        self.scanners = [
            filescan.PoolScanFile,
            filescan.PoolScanDriver,
            filescan.PoolScanSymlink,
            filescan.PoolScanMutant,
            filescan.PoolScanProcess,
            modscan.PoolScanModule,
            modscan.PoolScanThread,
            atoms.PoolScanAtom,
            windowstations.PoolScanWind,
            ]
        addr_space = utils.load_as(self._config)
        version = (addr_space.profile.metadata.get("major", 0),
                   addr_space.profile.metadata.get("minor", 0))
        if version < (6, 0):
            self.scanners.append(sockscan.PoolScanSocket)
            self.scanners.append(connscan.PoolScanConn)
        else:
            self.scanners.append(netscan.PoolScanUdpEndpoint)
            self.scanners.append(netscan.PoolScanTcpListener)
            self.scanners.append(netscan.PoolScanTcpEndpoint)
            self.scanners.append(callbacks.PoolScanDbgPrintCallback)
            self.scanners.append(callbacks.PoolScanRegistryCallback)
            self.scanners.append(callbacks.PoolScanPnp9)
            self.scanners.append(callbacks.PoolScanPnpD)
            self.scanners.append(callbacks.PoolScanPnpC)
        self.scanners.append(callbacks.PoolScanFSCallback)
        self.scanners.append(callbacks.PoolScanShutdownCallback)
        self.scanners.append(callbacks.PoolScanGenericCallback)
        for obj in self.scan_results(addr_space):
            if obj.size() != None:
                #dict_addr_to_type[obj.obj_offset + 0xffff000000000000] = obj.obj_type
                dict_addr_to_type[obj.obj_offset] = obj.obj_type
                dict_type_to_size[obj.obj_type] = obj.size()

        # add process info
        list_process_addr, process_size = self.process_scan()
        for addr in list_process_addr:
            #dict_addr_to_type[addr + 0xffff000000000000] = '_EPROCESS'
            dict_addr_to_type[addr] = '_EPROCESS'
        dict_type_to_size['_EPROCESS'] = process_size

        return dict_addr_to_type, dict_type_to_size

    def calculate(self):
        image_name = os.path.basename(self._config.LOCATION)
        #print image_name
        self.kernel_address_space = utils.load_as(self._config)

        dict_addr_to_type, dict_type_to_size = self.multiscan()
        addrs = dict_addr_to_type.keys()
        addrs.sort()
        if VIRTUAL == True:
            file_name = OUTPUT_PATH + 'label.' + image_name + '.multiscan'
        else:
            file_name = OUTPUT_PATH + 'label.' + image_name + '.multiscan.physical'
        with open(file_name, 'w') as f:
            for addr in addrs:
                f.write(hex(addr) + '\t' + dict_addr_to_type[addr] + '\t' + str(dict_type_to_size[dict_addr_to_type[addr]]) + '\n')

    def render_text(self, outfd, data):
        return
