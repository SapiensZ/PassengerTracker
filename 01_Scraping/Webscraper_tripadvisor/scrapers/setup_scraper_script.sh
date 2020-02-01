cd /usr/bin/
ls
sudo yum install python36
alternatives --set python /usr/bin/python3.6
python --version
cd /tmp
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip3 --version
pip3 install selenium --user
pip3 install bs4 --user
pip3 install pandas --user
pip3 install pyyaml --user

cd/tmp/
wget https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/bin/chromedriver
chromedriver --version


curl https://intoli.com/install-google-chrome.sh | bash
sudo mv /usr/bin/google-chrome-stable /usr/bin/google-chrome
google-chrome --version && which google-chrome

cd 



#python3 scraping.py &