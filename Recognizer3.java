package com.emaraic.ObjectRecognition;

import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import javax.imageio.ImageIO;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;


/**
 * Created with IntelliJ IDEA.
 * User: Vishnu Prem
 * Date: 2/25/18
 * Time: 10:50 PM
 * To change this template use File | Settings | File Templates.
 */

/**
 *
 * @author Taha Emara
 * Website: http://www.emaraic.com
 * Email : taha@emaraic.com
 * Created on: Apr 29, 2017
 * Kindly: Don't remove this header
 * Download the pre-trained inception model from here: https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
 */

public class Recognizer3 {

    private static byte[] graphDef;
    private static List<String> labels;

    private static String modelpath;
    private static String imagepath;

    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {                             //predict
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1)
                {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];

                float[][] res = new float[1][nlabels];       ///ADDED
                result.copyTo(res);                           //ADDED
                return res[0];                                //ORIGINAL: return result.copyTo(new float[1][labels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {                                            // find max label
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {                                           // read graph
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {                                     // reads labels file
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }



    public static void main(String[] args) {

        modelpath = "C:\\Users\\Vishnu Prem\\Desktop\\Inception java\\inception";
        graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
        labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));


        Scanner input = new Scanner(System.in);

        while(true){
             System.out.printf("\n\nEnter image name: ");
             String imagename = input.next();

            imagepath =  "C:\\Users\\Vishnu Prem\\Desktop\\Inception java\\test images\\" + imagename + ".jpg";

            try{
                    Image img = ImageIO.read(new File(imagepath));
            }
            catch(IOException e){
                System.out.println("Enter valid image name!");
                continue;
            }

            long starttime = System.nanoTime();

            byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));

            try (Tensor image = Tensor.create(imageBytes)) {                                          //IMP
                float[] labelProbabilities = executeInceptionGraph(graphDef, image);                  //IMP
                int bestLabelIdx = maxIndex(labelProbabilities);                                      //IMP
                System.out.println(String.format("Image identified as  %s (%.2f%%)",labels.get(bestLabelIdx).toUpperCase(), labelProbabilities[bestLabelIdx] * 100f));     //IMP
            }

            System.out.printf("Elapsed time = %s seconds", (System.nanoTime()-starttime)/1000000000.0);
        }
    }
}
