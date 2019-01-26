package util;

import java.security.MessageDigest;

/**
 * 正则工具包，注意：在写分组时，如果不需要后向引用（重复的情况，如go go）,要加(?:xxxx),可以减少内存消耗
 * 
 * @author 龚帅宾
 *
 */
public class RegexUtils {
	public static String cleanParaAndImgLabel(String raw) {
		return cleanSpecialWord(raw.replaceAll("\\$#imgidx=\\d{4}#\\$", "").replaceAll("!@#!@", ""));
	}
	
	public static String cleanSpecialWord(String text) {
		String regex = "\\s+|　+|&nbsp;+| +|[\\u0000]+|(?:\\r)+|(?:\\n)+|[\\u2003]+|[\\u3000]+|　+| +";
		return text.replaceAll(regex, "");
	}
    public static String md5Encode(String inStr) throws Exception {
        MessageDigest md5 = null;
        try {
            md5 = MessageDigest.getInstance("MD5");
        } catch (Exception e) {
            System.out.println(e.toString());
            e.printStackTrace();
            return "";
        }

        byte[] byteArray = inStr.getBytes("UTF-8");
        byte[] md5Bytes = md5.digest(byteArray);
        StringBuffer hexValue = new StringBuffer();
        for (int i = 0; i < md5Bytes.length; i++) {
            int val = ((int) md5Bytes[i]) & 0xff;
            if (val < 16) {
                hexValue.append("0");
            }
            hexValue.append(Integer.toHexString(val));
        }
        return hexValue.toString();
    }
}
