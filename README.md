Skymind SFU Coding Exercise
=========================
[See Details](https://www.zepl.com/viewer/notebooks/bm90ZTovL2Nyb2NrcG90dmVnZ2llcy9hZjAyZmEzOTk3M2Y0NmRhODFhM2Y0OGMzNmU0OTI5NC9ub3RlLmpzb24)
- Implementated as Caltech101 in dl4j-examples  
![dl4j-caltech101](https://github.com/nilichen/dl4j-examples/blob/master/dl4j-examples/src/main/resources/readme_images/dl4j-caltech101.png)
- Applied transfer learning with VGG-16, modifying only the last layer, keeping other frozen. See https://deeplearning4j.org/transfer-learning and https://deeplearning4j.org/build_vgg_webapp.     
![scores](https://github.com/nilichen/dl4j-examples/blob/master/dl4j-examples/src/main/resources/readme_images/score.png)

---

## Build and Run
**Run locally**
- `$ git clone https://github.com/nilichen/dl4j-examples.git` and follow deeplearning4j [Quick Start Guide](https://deeplearning4j.org/quickstart)
- For batchsize of 16 on MacBook Pro with 16GB memory, need to configure VM parameters in IntelliJ as follows, see https://deeplearning4j.org/benchmark
![VM-config](https://github.com/nilichen/dl4j-examples/blob/master/dl4j-examples/src/main/resources/readme_images/VM-config.png)


**Run on AWS with CUDA**
- Launch instance, choose Deep Learning AMI CUDA 8 with p2.xlarge, set storage as 100GB and make sure security group allows inbound SSH traffic from my public IPv4 address
<!-- ![launch-instance](images/launch-instance.png)     -->
- Install Sun Java 8 SDK as it comes with JavaFX and set it as the default java.
```
$ sudo wget \
        --no-cookies --header "Cookie: gpw_e24=xxx; oraclelicense=accept-securebackup-cookie;" \
        "http://download.oracle.com/otn-pub/java/jdk/8u151-b12/e758a0de34e24606bca991d704f6dcbf/jdk-8u151-linux-x64.rpm"
$ sudo rpm -i jdk-8u151-linux-x64.rpm
$ sudo /usr/sbin/alternatives --config java
$ sudo /usr/sbin/alternatives --config javac
```
Add the export of the java home to .bash_profile `export JAVA_HOME=/usr/java/default`  
```
$ source ~/.bash_profile
```
- Install Maven.
```
$ sudo wget \
        http://repos.fedorapeople.org/repos/dchen/apache-maven/epel-apache-maven.repo \
        -O /etc/yum.repos.d/epel-apache-maven.repo
$ sudo sed -i s/\$releasever/6/g /etc/yum.repos.d/epel-apache-maven.repo
$ sudo yum install -y apache-maven
$ mvn --version
```
- In general follow the [Quick Start Guide](https://deeplearning4j.org/quickstart) and [Using the Command Line on AWS](https://deeplearning4j.org/gettingstarted#advanced-using-the-command-line-on-aws) with minor changes
```
$ git clone https://github.com/nilichen/dl4j-examples.git  
$ cd dl4j-examples
```
- Make change to pom.xml to enable GPU. See https://deeplearning4j.org/gpu
- See https://deeplearning4j.org/memory for configuration
```
$ mvn clean install
$ cd dl4j-examples
$ java -cp target/dl4j-examples-0.9.1-bin.jar \
         org.deeplearning4j.examples.Caltech101.Caltech101Classification \
        -Xmx6G -Dorg.bytedeco.javacpp.maxbytes=12G \
        -Dorg.bytedeco.javacpp.maxphysicalbytes=12G
```
