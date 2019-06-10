import os
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

def get_credentials(colab=False):
    if colab:
        from google.colab import auth

        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()

    else:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()

    return  GoogleDrive(gauth)


def downloader(drive, set_name, drive_id, path='data/'):
    #  https://developers.google.com/drive/v2/web/search-parameters
    file_list = drive.ListFile(
        {'q': "'%s' in parents" % drive_id}).GetList()

    print('downloading %s set' % set_name)
    for f in tqdm(file_list):
        fname = f['title']
        f_ = drive.CreateFile({'id': f['id']})
        f_.GetContentFile(path + set_name + '/' + fname)

if __name__=='__main__':
    try:
        os.makedirs('data/train')
        os.makedirs('data/test')
    except: pass

    drive = get_credentials(colab=True)
    downloader(drive, 'train', '15_I4PO9mYgNsSlHpk8hbYBdg011xhC2e')
    downloader(drive, 'test', '1qSD5Yu-BqF4iF1GOaR9EzFA87vAIrQny')
