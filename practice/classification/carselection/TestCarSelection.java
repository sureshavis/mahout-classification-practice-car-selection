package practice.classification.carselection;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.StringReader;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import com.google.common.base.Charsets;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

public class TestCarSelection {
	private final static Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_36);
	private final static FeatureVectorEncoder vectorEncoder = new StaticWordValueEncoder("body");
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		if(args == null || args.length == 0){
			System.out.println("no arguments found..");
		}else{
			String inputFilePath = args[0];
			String modelFilePath = args[1];
			String currentLine   = null;
			OnlineLogisticRegression regressionClassifier = ModelSerializer.readBinary(new FileInputStream(modelFilePath), OnlineLogisticRegression.class);
			Dictionary wordDictionary = new Dictionary();
			Multiset<String> wordCount = ConcurrentHashMultiset.create();
			List<String> wordEntries = Lists.newArrayList();
			BufferedReader fileReader = null;
			try{
				
				System.out.println("reading the file contents...");
				fileReader = Files.newReader( new File(inputFilePath), Charsets.UTF_8);

				while( (currentLine = fileReader.readLine()) != null){
					if(!currentLine.startsWith("@")){
						TokenStream tokenStream =  analyzer.tokenStream(null, new StringReader(currentLine));
						while(tokenStream.incrementToken()){}
						wordDictionary.intern(tokenStream.getAttribute(CharTermAttribute.class).toString());
					}
				}	
				
				ResultAnalyzer resultAnalyzer = new ResultAnalyzer(wordDictionary.values(), "DEFAULT");
				fileReader =  Files.newReader( new File(inputFilePath), Charsets.UTF_8);
				
				while( (currentLine = fileReader.readLine()) != null){
					if(!currentLine.startsWith("@")){
						TokenStream tokenStream =  analyzer.tokenStream(null, new StringReader(currentLine));
						//System.out.println("tokenizing the contents of the current line..");
						while(tokenStream.incrementToken()){
							String token = tokenStream.getAttribute(CharTermAttribute.class).toString();
							wordEntries.add(token);
						}
						tokenStream.clearAttributes();
						tokenStream.close();
						
						for(int index = 0; index < wordEntries.size()-2; index++){
							wordCount.add(wordEntries.get(index));
						}
						String entryLabel = wordEntries.get(wordEntries.size()-1);
						
						//System.out.println("registering the label with the dictionary...");
						int dictionaryIndex = wordDictionary.intern(entryLabel);
						
						//System.out.println("creating the vector for the entries..");
						Vector entryVector = new RandomAccessSparseVector(6);
						for(String word : wordCount.elementSet()){
							vectorEncoder.addToVector(word, Math.log1p(wordCount.count(word)), entryVector);
						}
						wordEntries.clear();
						
						Vector resultVector = regressionClassifier.classify(entryVector);
						int maxValueIdx = resultVector.maxValueIndex();
						double maxValue = resultVector.maxValue();
						double logLHood = regressionClassifier.logLikelihood(dictionaryIndex, resultVector);
						ClassifierResult classifierResult = new ClassifierResult(wordDictionary.values().get(maxValueIdx), maxValue, logLHood);
						System.out.println(entryLabel+"==>"+ classifierResult.getLabel());
					}
				}
			}catch(Exception e){
				System.out.println("exception occured "+e);
			}finally{
				Closeables.close(fileReader,true);
			}
			
		}			
		
	}
}
