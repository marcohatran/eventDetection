package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class NewsProducers extends Thread {
	private volatile boolean isRunning = true;
	private BlockingQueue<String> queue;// 内存缓冲区
	private static AtomicInteger count = new AtomicInteger();// 总数 原子操作
	public NewsProducers(BlockingQueue<String> queue) {
		this.queue = queue;
	}

	@Override
	public void run() {
		System.out.println("start producting id:" + Thread.currentThread().getId());
		try {
			Random r = new Random();
			InputStream is = null;
			BufferedReader br = null;
			try {
				is = new FileInputStream(new File("D:\\duplicate\\split\\part_100000"));
				br = new BufferedReader(new InputStreamReader(is, "utf-8"));
				String s = null;
				while ((s = br.readLine()) != null) {
					queue.put(s.trim());
					Thread.sleep(r.nextInt(100));
					System.out.println(count.incrementAndGet());
				}
			} catch (FileNotFoundException e) {
			} catch (IOException e) {
			} finally {
				try {
					if (br != null)
						br.close();
				} catch (IOException e) {
				}
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
			Thread.currentThread().interrupt();
		}

	}
}
