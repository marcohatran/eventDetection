package util;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Properties;
import java.util.PropertyResourceBundle;
import java.util.ResourceBundle;

import org.nlpcn.commons.lang.util.FileFinder;
import org.nlpcn.commons.lang.util.IOUtil;

public class LoadConf {
	private ResourceBundle rb = null;

	private LoadConf() {
		if (rb == null) {
			try {
				rb = ResourceBundle.getBundle("dictionary");
			} catch (Exception e) {
				try {
					File find = FileFinder.find("dictionary.properties", 2);
					if (find != null && find.isFile()) {
						rb = new PropertyResourceBundle(
								IOUtil.getReader(find.getAbsolutePath(), System.getProperty("file.encoding")));
						System.out.println("load library not find in classPath ! i find it in " + find.getAbsolutePath()
								+ " make sure it is your config!");
					}
				} catch (Exception e1) {
					System.out
							.println(String.format("not find dictionary.properties. and err {} i think it is a bug!", e1));

				}
			}
		}
	}

	public static LoadConf getIstance() {
		if (instance == null) {
			synchronized (LoadConf.class) {
				if (instance == null) {
					instance = new LoadConf();
				}
			}
		}
		return instance;
	}

	private static volatile LoadConf instance;

	public boolean chkProperty(String _key) {
		return rb.containsKey(_key);
	}

	public String getProperty(String _key) {
		return rb.getString(_key);
	}
}
