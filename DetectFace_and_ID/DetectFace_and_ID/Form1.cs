using Emgu;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DetectFace_and_ID
{
    public partial class Form1 : Form
    {
        Capture capture;
        Image<Bgr, byte> image;
        int facesNumber;
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(capture == null)
            {
                //here i'am using defualt camera
                capture = new Capture(0);
            }
            capture.ImageGrabbed += Capture_ImageGrabbed;
            capture.Start();
            //Draw rectangle arround the id
            image = DetectID(image);
            //Draw rectangle arround the face
            image = DetectFace(image);
            //disply the image
            imageBox1.Image = image;
            //save the image in the data folder
            image.Save("ms-appx:///data/output.jpg");
            //number of faces in the image
            MessageBox.Show(string.Concat("number of faces in this image is :", facesNumber.ToString()));
        }

        private Image<Bgr, byte> DetectFace(Image<Bgr, byte> im)
        {
            try
            {
                if(im == null)
                {
                    return im;
                }
                Uri facePath = new Uri("ms-appx:///data/haarcascade_frontalface_default.xml");
                CascadeClassifier cascadeClassifier = new CascadeClassifier(facePath.AbsolutePath);
                var imgGray = im.Convert<Gray, byte>().Clone();
                Rectangle[] faces = cascadeClassifier.DetectMultiScale(im, 1.1, 4);
                facesNumber = faces.Length;
                foreach(var face in faces)
                {
                    im.Draw(face, new Bgr(0, 0, 255), 2);
                }
                return im;
            }catch(Exception ex)
            {
                MessageBox.Show(ex.Message);
                return null;
            }
        }
        private void Capture_ImageGrabbed(object sender, EventArgs e)
        {
            try
            {
                Mat mat = new Mat();
                capture.Retrieve(mat);
                image = mat.ToImage<Bgr, byte>();
                imageBox1.Image = image;
            }
            catch (Exception)
            {

            }
        }

        private Image<Bgr, byte> DetectID(Image<Bgr, byte> im)
        {
            if (im == null)
                return im;
            try
            {
                var temp = im.SmoothGaussian(5).Convert<Gray, byte>().ThresholdBinaryInv(new Gray(230), new Gray(255));

                VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
                Mat m = new Mat();
                CvInvoke.FindContours(temp, contours, m, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                for(int i = 0; i < contours.Size; i++)
                {
                    double perimeter = CvInvoke.ArcLength(contours[i], true);
                    VectorOfPoint approx = new VectorOfPoint();
                    CvInvoke.ApproxPolyDP(contours[i], approx, 0.04 * perimeter, true);
                    CvInvoke.DrawContours(im, contours, i, new MCvScalar(0, 0, 255), 2);
                    var moments = CvInvoke.Moments(contours[i]);
                    int x = (int)(moments.M10 / moments.M00);
                    int y = (int)(moments.M01 / moments.M00);

                    if(approx.Size == 4)
                    {
                        Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                        double ratio = rect.Width / rect.Height;
                        if(ratio > 1.5 && ratio < 1.6)
                        {
                            CvInvoke.DrawContours(im, contours, i, new MCvScalar(0, 0, 255), 2);
                        }
                            
                    }
                }
                return im;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
                return null;
            }
        }
    }
}
