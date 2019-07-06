import os
import sys
from time import gmtime, strftime

MAX = 6
PROFILE='Win7SP1x86'

def get_file_idx(dump_name):
    file_idx = dump_name.replace('memdump_win7_', '').replace('.raw','')
    if '_' in file_idx:
        s = file_idx.split('_')
        file_idx = 200 + int(s[1])
    else:
        file_idx = int(file_idx)
    return file_idx
    
def log(message):
    print('%s\t%s' %(strftime("%Y-%m-%d %H:%M:%S", gmtime()), message))
    sys.stdout.flush()

def main():
    image_path = '../memory_dumps/'
    image_files = os.listdir(image_path)
    image_files.sort()
    os.system('mkdir -p label pages')
    label_files = os.listdir('./label/')
    page_files = os.listdir('./pages/')
    for dump_name in image_files:
        file_idx = get_file_idx(dump_name)
        log(dump_name)
        cmd_label_multiscan = 'cd ../volatility; python vol.py -f ' + image_path + dump_name + ' --profile=' + PROFILE + ' my_multiscan -V; cd -'
        if 'label.' + dump_name + '.multiscan' not in label_files:
            log(cmd_label_multiscan)
            os.system(cmd_label_multiscan)
        cmd_label_handles = 'cd ../volatility; python vol.py -f ' + image_path + dump_name + ' --profile=' + PROFILE + ' my_handles -V; cd -'
        if 'label.' + dump_name + '.handles' not in label_files:
            log(cmd_label_handles)
            os.system(cmd_label_handles)
        cmd_get_pages = 'cd ../volatility; python vol.py -f ' + image_path + dump_name + ' --profile=' + PROFILE + ' get_physical_pages; cd -'
        if 'pages.' + dump_name not in page_files:
            log(cmd_get_pages)
            os.system(cmd_get_pages)
        cmd_build_graph = 'python graph.py ' + image_path + dump_name
        log(cmd_build_graph)
        os.system(cmd_build_graph)

if __name__ == "__main__":
    main()
