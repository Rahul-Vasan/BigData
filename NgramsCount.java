package edu.nyu.bigdata;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.*;




public class NgramsCount {


    enum TYPES {
        UNI, BI, TRI;
    };


    public static void main(String[] args) throws Exception {


        Configuration conf = new Configuration();


        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length <= 1) {
            System.err.println("Usage: ngramcount <in> <out>");
            System.exit(2);
        }


        Job job = new Job(conf, "job1: Computing the global Ngram counts");


        job.setJarByClass(NgramsCount.class);

        job.setMapperClass(Mapper1.class);
        job.setReducerClass(Reducer1.class);


        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);


        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));


        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1].concat("-job1")));

        boolean status = job.waitForCompletion(true);


        long uni_count = job.getCounters().findCounter(TYPES.UNI).getValue();
        long bi_count = job.getCounters().findCounter(TYPES.BI).getValue();
        long tri_count = job.getCounters().findCounter(TYPES.TRI).getValue();


        if (status) {

            conf = new Configuration();


            conf.set("UNICOUNT",String.valueOf(uni_count));
            conf.set("BICOUNT",String.valueOf(bi_count));
            conf.set("TRICOUNT",String.valueOf(tri_count));


            job = new Job(conf, "Job 2: Computing the dividend and calculating the probability ");
            job.setJarByClass(NgramsCount.class);

            job.setMapperClass(Mapper2.class);
            job.setReducerClass(Reducer2.class);


            FileInputFormat.addInputPath(job, new Path(String.format(otherArgs[1].concat("-job1/part-*"))));


            FileOutputFormat.setOutputPath(job, new Path(otherArgs[1].concat("-job2")));


            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);


            job.setNumReduceTasks(1);

            status = job.waitForCompletion(true);
        }
        if (status) {

            conf = new Configuration();


            job = new Job(conf, "Job 3: Computing Conditional Probability");
            job.setJarByClass(NgramsCount.class);

            job.setMapperClass(Mapper3.class);
            job.setReducerClass(Reducer3.class);


            FileInputFormat.addInputPath(job, new Path(String.format(otherArgs[1].concat("-job2/part-*"))));


            FileOutputFormat.setOutputPath(job, new Path(otherArgs[1].concat("-job3")));


            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);


            job.setNumReduceTasks(1);

            status = job.waitForCompletion(true);
        }

        System.exit( status? 0 : 1);
    }


    public static class Mapper1 extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();


        List<String> tokens = new ArrayList<String>();

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {


            tokens.clear();
            String singleline = value.toString().toLowerCase().replaceAll("[^a-z0-9]", " ");




            StringTokenizer itr = new StringTokenizer(singleline);
            while (itr.hasMoreTokens()) {
                tokens.add(itr.nextToken());
            }


            int length = tokens.size();
            int i =0;
            while(i<length)
            {
                word.set(tokens.get(i));

                context.write(word, one);

                if (i < length - 1) {
                    word.set(String.format("%s %s", tokens.get(i), tokens.get(i + 1)));

                    context.write(word, one);
                }
                if (i < length - 2) {
                    word.set(String.format("%s %s %s", tokens.get(i), tokens.get(i + 1), tokens.get(i + 2)));

                    context.write(word, one);
                }
                i++;
            }
        }
    }


    public static class Reducer1 extends Reducer<Text,IntWritable,Text,IntWritable> {

        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {


            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);

            String[] type = key.toString().split(" ");
            if (type.length == 1)
                context.getCounter(TYPES.UNI).increment(1);
            else if (type.length == 2)
                context.getCounter(TYPES.BI).increment(1);
            else if (type.length == 3)
                context.getCounter(TYPES.TRI).increment(1);

        }
    }

    public static class Mapper2 extends Mapper<Object, Text, Text, Text> {


        private static final Text UNI = new Text("UNI");
        private static final Text BI = new Text("BI");
        private static final Text TRI = new Text("TRI");

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            int length = value.toString().split("[ \t]").length - 1;
            if (length == 1)
                context.write(UNI, value);
            else if (length == 2)
                context.write(BI, value);
            else if (length == 3)
                context.write(TRI, value);
        }
    }



    public static class Reducer2 extends Reducer<Text,Text,Text,Text> {

        private Text result = new Text();
        private Text ngram = new Text();

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {

            String divisor = null;

            if (key.toString().compareTo("UNI")==0) {
                divisor = context.getConfiguration().get("UNICOUNT");

            }
            else if (key.toString().compareTo("BI")==0) {
                divisor = context.getConfiguration().get("BICOUNT");

            }
            else if (key.toString().compareTo("TRI")==0) {
                divisor = context.getConfiguration().get("TRICOUNT");

            }




            float probability;
            for (Text val : values) {
                probability =0;
                String[] words = val.toString().split("\t");
                if (words.length>1) {

                    ngram.set(words[0]);


                    float dividend = Float.parseFloat(words[1]);
                    float divisor1 = Float.parseFloat(divisor);
                    probability = dividend/divisor1;

                    result.set(String.valueOf(probability));
                    context.write(ngram, result);

                }
            }
        }
    }
    public static class Mapper3 extends Mapper<Object, Text, Text, Text> {


        private static final Text TRI = new Text("TRI");

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {


            int len = value.toString().split("[ \t]").length - 1;
            if (len==3) {
                String[] tokens = value.toString().split("[ \t]");
                if (tokens[0].equals("united") && tokens[1].equals("states")) context.write(TRI, value);
            }
        }
    }

    public static class Reducer3 extends Reducer<Text,Text,Text,Text> {

        private Text word = new Text();
        private Text probability = new Text();
        float curr_max = 0;
        String curr_max_word ="";

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException,NullPointerException {


            for (Text val : values) {

                String[] words = val.toString().split("\t");
                float prob = Float.parseFloat(words[1]);
                if(prob>curr_max)
                {
                    curr_max = prob;
                    curr_max_word = words[0];
                }

            }
            word.set(curr_max_word);
            probability.set(String.valueOf(curr_max));
            context.write(word,probability);

        }
    }
}



