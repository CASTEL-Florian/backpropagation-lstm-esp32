/* THIS IS THE MAIN FOR AN 32 BIT LSTM */


#include "freertos/FreeRTOS.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "freertos/task.h"

#include <math.h>
#include "time.h"
#define PI 3.141592654
//#include "parameters32.h"
#include "lstm.h"

float time_series[TIME_SERIES_LEN] = {0.36800376, 0.44377467, 0.44579864, 0.4892158, 0.41918758, 0.43368587, 0.4195563, 0.45447323, 0.52435946, 0.35436207, 0.45200756, 0.42453113, 0.4379751, 0.45101032, 0.42535964, 0.43436086, 0.44679898, 0.43804312, 0.42917702, 0.45167574, 0.43036568, 0.43240765, 0.45368606, 0.54604393, 0.3630159, 0.41708392, 0.454731, 0.42406732, 0.44269663, 0.4644018, 0.41911578, 0.4519673, 0.45940274, 0.41718924, 0.44855547, 0.4584904, 0.4292666, 0.4626228, 0.4499618, 0.43687418, 0.4638575, 0.4655049, 0.4300672, 0.45319834, 0.45886976, 0.44951713, 0.45615074, 0.4915811, 0.46147013, 0.45644698, 0.40435022, 0.4537275, 0.45876446, 0.46432948, 0.4284916, 0.44889468, 0.48743322, 0.3798379, 0.41644782, 0.45052013, 0.569902, 0.5086686, 0.337551, 0.41551748, 0.4522733, 0.49720466, 0.4324856, 0.40151435, 0.4334952, 0.44803065, 0.50311613, 0.45939445, 0.3530923, 0.46985042, 0.42696294, 0.520422, 0.41582397, 0.44421077, 0.4040704, 0.48347378, 0.39063263, 0.45571056, 0.47868747, 0.44030064, 0.47118387, 0.4231816, 0.5724622, 0.31677997, 0.47210026, 0.49203032, 0.40036944, 0.37709683, 0.44208226, 0.4688486, 0.41500318, 0.44460303, 0.41473216, 0.45754477, 0.43863857, 0.521662, 0.3797822, 0.46163583, 0.48336923, 0.39384553, 0.4235066, 0.50196594, 0.4403816, 0.4015379, 0.45553514, 0.48299053, 0.49378017, 0.37154892, 0.44855955, 0.40978396, 0.46394724, 0.42241693, 0.47361445, 0.5058676, 0.37122676, 0.44311598, 0.49533734, 0.46195215, 0.43598115, 0.43920252, 0.3714337, 0.4779872, 0.45170233, 0.40239948, 0.51398385, 0.3616156, 0.42558146, 0.55841243, 0.3655607, 0.42975035, 0.43041545, 0.44251946, 0.4281253, 0.46803474, 0.4633927, 0.38991833, 0.44595036, 0.42997092, 0.4480702, 0.42294124, 0.4951438, 0.44612032, 0.40607572, 0.39445662, 0.45257485, 0.42382917, 0.45040584, 0.42654288, 0.4454346, 0.43402755, 0.45201454, 0.4416478, 0.43861127, 0.43781823, 0.4472117, 0.45068127, 0.42456767, 0.4327148, 0.46710536, 0.41861698, 0.4534196, 0.4502921, 0.42438862, 0.4508893, 0.44754228, 0.4275483, 0.44832444, 0.45372725, 0.43511456, 0.4431639, 0.44296294, 0.4435963, 0.45163286, 0.4414828, 0.4352084, 0.45590687, 0.44411394, 0.4515704, 0.48297036, 0.4313181, 0.44498217, 0.46248204, 0.4595731, 0.58098686, 0.30955935, 0.4658548, 0.47224224, 0.4524894, 0.5021428, 0.4864661, 0.40316743, 0.4702524, 0.44571427, 0.44908434, 0.46673527, 0.41433898, 0.5139425, 0.3864617, 0.41616744, 0.39264825, 0.552463, 0.40262353, 0.50951743, 0.39505118, 0.4467095, 0.42612696, 0.48001114, 0.35964158, 0.4210091, 0.4763076, 0.4286005, 0.4508372, 0.63229454, 0.30742145, 0.3781769, 0.47730923, 0.455915, 0.47804114, 0.45794868, 0.37581545, 0.46783757, 0.41270274, 0.41843817, 0.46852368, 0.4594698, 0.5105381, 0.47822133, 0.37850747, 0.45928222, 0.4526488, 0.37759373, 0.468281, 0.3900751, 0.46746686, 0.51584345, 0.38932094, 0.4627149, 0.55922663, 0.42623922, 0.4083818, 0.46404806, 0.4422177, 0.3683376, 0.41483605, 0.4306703, 0.46606344, 0.38289636, 0.44567218, 0.48901612, 0.4291786, 0.45491377, 0.44902036, 0.41351697, 0.47317198, 0.47340032, 0.3906534, 0.46244448, 0.5004776, 0.47455683, 0.3197381, 0.53770983, 0.37146103, 0.44922367, 0.4475563, 0.5292829, 0.37561205, 0.4070682, 0.40207723, 0.48437726, 0.42931232, 0.46501645, 0.46464577, 0.39131856, 0.4010512, 0.45388728, 0.6262115, 0.28491724, 0.42484036, 0.4913472, 0.3927414, 0.49710163, 0.40079516, 0.40904102, 0.4557483, 0.42595127, 0.44996944, 0.4204593, 0.44631913, 0.43203613, 0.47212404, 0.41782567, 0.43164235, 0.457347, 0.43825167, 0.4296117, 0.4284404, 0.45693892, 0.44422984, 0.45132273, 0.43075287, 0.4484212, 0.42383012, 0.45811257, 0.506441, 0.41079667, 0.4087205, 0.43120265, 0.43974537, 0.45588616, 0.44258362, 0.42390957, 0.45816335, 0.4343789, 0.43953565, 0.44560158, 0.4410408, 0.4387668, 0.44953638, 0.44453037, 0.43480572, 0.4539758, 0.43589953, 0.44097102, 0.45467743, 0.44029558, 0.44666597, 0.47449905, 0.44205397, 0.4618121, 0.47116444, 0.45524785, 0.49902517, 0.45898724, 0.40123186, 0.5405023, 0.38559857, 0.46034503, 0.47375262, 0.4258361, 0.4636211, 0.4359966, 0.54315144, 0.3523213, 0.41914546, 0.4667136, 0.48861688, 0.39797288, 0.48132932, 0.55836916, 0.48878333, 0.39306605, 0.352282, 0.38190827, 0.4386452, 0.47004876, 0.48041475, 0.43324056, 0.39578488, 0.41345823, 0.5342507, 0.43398583, 0.37255445, 0.47345018, 0.4298702, 0.4747876, 0.48912138, 0.3743782, 0.45862278, 0.4732597, 0.40640232, 0.42692372, 0.46697882, 0.44773683, 0.46341, 0.41406366, 0.42525735, 0.49988872, 0.4172013, 0.47386903, 0.37721473, 0.46859986, 0.4834733, 0.38538575, 0.43993834, 0.45095205, 0.45235187, 0.50248355, 0.3806375, 0.40206423, 0.4837141, 0.46284455, 0.44149786, 0.4588833, 0.4309229, 0.51585525, 0.36854684, 0.54391205, 0.38014477, 0.43938747, 0.48886248, 0.33870244, 0.53420323, 0.3743764, 0.43307167, 0.45570856, 0.5159432, 0.40519258, 0.5149679, 0.35838324, 0.36996493, 0.51086974, 0.45006847, 0.46026152, 0.41222736, 0.4458263, 0.4192402, 0.4526911, 0.38981554, 0.46042687, 0.39805913, 0.4796488, 0.44099265, 0.5151578, 0.36904302, 0.4414112, 0.44953507, 0.46407565, 0.4239787, 0.47927564, 0.40698627, 0.43363503, 0.42567092, 0.45078248, 0.3929547, 0.440268, 0.46221423, 0.42178822, 0.4481733, 0.44685823, 0.42013955, 0.44724658, 0.45469153, 0.43059608, 0.43255144, 0.46693563, 0.4197447, 0.4299964, 0.47096103, 0.43029937, 0.42272225, 0.4622984, 0.42866212, 0.44154733, 0.45180878, 0.43846497, 0.4383125, 0.44697994, 0.43919384, 0.44514555, 0.44476312, 0.43029886, 0.43945155, 0.4568275, 0.45168266, 0.43383217, 0.48028415, 0.42288604, 0.4369724, 0.4737599, 0.46654943, 0.46095395, 0.47113624, 0.42084238, 0.48500067, 0.5146164, 0.3648502, 0.5235784, 0.4727701, 0.42402545, 0.3899481, 0.5192893, 0.4053011, 0.44497168, 0.4626183, 0.4496485, 0.42901808, 0.48809183, 0.41174763, 0.40732896, 0.4952255, 0.4180033, 0.46439123, 0.41590953, 0.42955685, 0.47354865, 0.46408567, 0.37102142, 0.43889508, 0.44557095, 0.48652026, 0.4292288, 0.4424655, 0.45612055, 0.46365097, 0.43878835, 0.44890732, 0.423804, 0.47149774, 0.43612656, 0.39683738, 0.47760516, 0.44866067, 0.4694906, 0.41648477, 0.43313268, 0.4644654, 0.46935588, 0.42245436, 0.4393481, 0.43155548, 0.5553527, 0.33283573, 0.42893922, 0.49112344, 0.45746538, 0.427414, 0.41526154, 0.4574931, 0.482773, 0.9, 0.1594899, 0.33728504, 0.32434404, 0.44359946, 0.45503795, 0.42208263, 0.5033945, 0.389896, 0.43311608, 0.50004244, 0.4054579, 0.45547858, 0.39270788, 0.4579959, 0.46305475, 0.40465903, 0.46805325, 0.41438305, 0.45378053, 0.42175674, 0.46059856, 0.46347147, 0.50126696, 0.31937408, 0.48006347, 0.44813645, 0.4431986, 0.43785337, 0.46721396, 0.44004154, 0.45174095, 0.434681, 0.44898677, 0.4648569, 0.46127808, 0.42632365, 0.3952323, 0.46491927, 0.43219632, 0.40473554, 0.44093654, 0.5198172, 0.36612597, 0.423649, 0.43629125, 0.45443794, 0.44970348, 0.4259112, 0.42406887, 0.4173271, 0.5525006, 0.3465315, 0.4541938, 0.518673, 0.35782066, 0.46794075, 0.41353843, 0.43169585, 0.5018974, 0.38668287, 0.47611138, 0.478931, 0.383882, 0.4715377, 0.39046532, 0.43038744, 0.4326817, 0.4436926, 0.4577018, 0.4233634, 0.45180354, 0.45602083, 0.42232856, 0.4674015, 0.44018897, 0.43185908, 0.44925648, 0.45366573, 0.42289713, 0.45848787, 0.44869903, 0.42400104, 0.47664884, 0.44690064, 0.43943688, 0.4847965, 0.43086872, 0.46629182, 0.528284, 0.42221853, 0.4861799, 0.48712564, 0.45350108, 0.3640158, 0.4794134, 0.4514153, 0.4322418, 0.4490716, 0.5052142, 0.5066679, 0.3991001, 0.3369655, 0.4766126, 0.43180418, 0.48478734, 0.4242833, 0.42871976, 0.51457316, 0.5928869, 0.31844813, 0.40968147, 0.4670346, 0.3490573, 0.4528514, 0.43435824, 0.4592836, 0.61477596, 0.2782786, 0.44491136, 0.4375399, 0.5135055, 0.39806947, 0.3731774, 0.47497052, 0.54558873, 0.44757932, 0.39489657, 0.5082883, 0.36850604, 0.49060324, 0.44359407, 0.40379605, 0.60017693, 0.3263598, 0.4153356, 0.53980696, 0.32552847, 0.44621226, 0.46162578, 0.421763, 0.38897514, 0.5571545, 0.38616446, 0.43861997, 0.44976002, 0.5775045, 0.28242376, 0.4398209, 0.4745785, 0.48428956, 0.49098128, 0.36364233, 0.44253722, 0.44125336, 0.41254753, 0.52568746, 0.62174726, 0.18021879, 0.46220058, 0.43772516, 0.5181667, 0.3875883, 0.51693225, 0.52166283, 0.25969395, 0.43806168, 0.5560905, 0.43289348, 0.3249439, 0.4351728, 0.4282631, 0.4270313, 0.5258384, 0.4510941, 0.36467677, 0.45569313, 0.49286538, 0.44938794, 0.4354471, 0.37966475, 0.42991266, 0.46373358, 0.4312399, 0.42615807, 0.4645045, 0.4379844, 0.45706075, 0.42549965, 0.42942086, 0.46402314, 0.4478937, 0.53455734, 0.3807973, 0.4043535, 0.42253086, 0.42174572, 0.4674987, 0.42542583, 0.44033998, 0.52935505, 0.46245438, 0.33399752, 0.43581417, 0.43043223, 0.4630834, 0.44034806, 0.41698524, 0.4645307, 0.44439802, 0.4133613, 0.45760155, 0.44682106, 0.4181961, 0.4665842, 0.43263307, 0.43211925, 0.46395403, 0.4292239, 0.43560544, 0.45329082, 0.43952635, 0.43664253, 0.45412615, 0.4490787, 0.44917688, 0.4339581, 0.45091304, 0.45071852, 0.51177645, 0.40578333, 0.45795634, 0.4292789, 0.5075186, 0.4587675, 0.5018768, 0.4104404, 0.41944423, 0.44502586, 0.4653385, 0.5295928, 0.40259942, 0.4299971, 0.5083239, 0.39316317, 0.49605033, 0.391373, 0.44110942, 0.4538063, 0.602352, 0.40013555, 0.33979028, 0.44184908, 0.56410116, 0.38875473, 0.31875366, 0.5595551, 0.43413174, 0.4038691, 0.40925607, 0.45826998, 0.42066252, 0.46039748, 0.41981775, 0.4833856, 0.40110493, 0.45883888, 0.44749978, 0.4150091, 0.49599764, 0.5262129, 0.47431585, 0.3429193, 0.4552051, 0.50223637, 0.40186325, 0.46775398, 0.42688555, 0.5115729, 0.5044259, 0.33302364, 0.49178687, 0.44745287, 0.8609139, 0.0, 0.48499882, 0.3729814, 0.34123394, 0.51817876, 0.5730432, 0.25803602, 0.4540059, 0.4563613, 0.3893063, 0.53164804, 0.39359125, 0.47633326, 0.43142992, 0.6569865, 0.46207792, 0.24403663, 0.44214588, 0.4783058, 0.3876131, 0.42325866, 0.4203478, 0.41013473, 0.4950913, 0.42903423, 0.4478974, 0.40985185, 0.48640674, 0.43230876, 0.4630583, 0.3475672, 0.46650127, 0.47374964, 0.5395654, 0.3390875, 0.4550595, 0.4297457, 0.41912368, 0.49590045, 0.36705834, 0.4350912, 0.48559988, 0.4830603, 0.52211684, 0.30011648, 0.42780164, 0.4804694, 0.5139563, 0.3361691, 0.41659793, 0.44836023, 0.45392388, 0.40290454, 0.47229972, 0.492212, 0.35946482, 0.44298393, 0.43859974, 0.43509513, 0.74413604, 0.21582548, 0.33963934, 0.537662, 0.4480694, 0.3411931, 0.5305421, 0.44410786, 0.3572353, 0.52763873, 0.44088542, 0.37605536, 0.48886916, 0.443951, 0.4086932, 0.4512615, 0.45187128, 0.41102496, 0.46423724, 0.4628521, 0.40966952, 0.46646482, 0.43739825, 0.44044316, 0.46291375, 0.44865647, 0.4184445, 0.47001353, 0.43705922, 0.447816, 0.46004292, 0.46153247, 0.45460486, 0.52657, 0.45715752, 0.44199917, 0.5102826, 0.41065258, 0.39011887, 0.5318561, 0.40337116, 0.47920665, 0.40025887, 0.46757042, 0.4805581, 0.41050717, 0.52505314, 0.50771517, 0.5248953, 0.2558005, 0.33700508, 0.45646316, 0.6412929, 0.40007788, 0.35319182, 0.6780274, 0.14678982, 0.47042403, 0.4959576, 0.4031804, 0.47670355, 0.39581123, 0.43676272, 0.43079442, 0.4579507, 0.5054384, 0.3983008, 0.5105208, 0.38950244, 0.40152675, 0.46321088, 0.46119112, 0.5292362, 0.44462562, 0.44126853, 0.3546761, 0.49975258, 0.47187755, 0.35325876, 0.46631807, 0.4373106, 0.4773241, 0.47831467, 0.54391026, 0.38625157, 0.46650705, 0.33923075, 0.621303, 0.45876634, 0.2857106, 0.4708258, 0.40208048, 0.4155392, 0.40950418, 0.4758278, 0.41519243, 0.39817977, 0.60313046, 0.32311627, 0.45942685, 0.45351708, 0.44784272, 0.43213, 0.4203248, 0.4360324, 0.4601131, 0.43582323, 0.41814536, 0.48721725, 0.42253932, 0.4074557, 0.43856066, 0.4412321, 0.55779415, 0.34134218, 0.40694875, 0.45607257, 0.54981875, 0.40905145, 0.67646945, 0.28932834, 0.3470115, 0.45326507, 0.46057644, 0.45943025, 0.38820902, 0.52966404, 0.3695356, 0.5043338, 0.3499054, 0.45224395, 0.4324679, 0.4939947, 0.37780464, 0.46193725, 0.44689187, 0.49609613, 0.37730166, 0.4217785, 0.47790533, 0.442393, 0.36463726, 0.46160617, 0.4564304, 0.40272155, 0.4537354, 0.42663968, 0.44014648, 0.4698027, 0.41533124, 0.44153565, 0.4649846, 0.40925175, 0.44054356, 0.4666763, 0.4173565, 0.45117164, 0.48741922, 0.43675554, 0.39414847, 0.46120548, 0.4475063, 0.41326156, 0.473079, 0.41961366, 0.45841068, 0.44485912, 0.4420363, 0.43045774, 0.48175955, 0.42798737, 0.43393448, 0.48441166, 0.4219944, 0.45480853, 0.44724712, 0.511838, 0.43249413, 0.42525396, 0.4697339, 0.45993146, 0.45089668, 0.4756381, 0.48116913, 0.47038236, 0.44786566, 0.4190191, 0.44538614, 0.4273104, 0.4320543, 0.4385596, 0.49910566, 0.39684382, 0.45009992, 0.4511784, 0.41571915, 0.4893786, 0.3828029, 0.45294216, 0.44201422, 0.48284993, 0.40138432, 0.46271425, 0.46685356, 0.37894195, 0.48101225, 0.48246846, 0.45327336, 0.46504343, 0.38210475, 0.44332993, 0.49187514, 0.522289, 0.298043, 0.45440584, 0.43444476, 0.5043403, 0.43548506, 0.39809138, 0.4222532, 0.52446306, 0.44173422, 0.41798267, 0.57606786, 0.557663, 0.32670704, 0.36442837, 0.39256683, 0.43283084, 0.4504811, 0.41455615, 0.5237485, 0.3971758, 0.4702177, 0.44560066, 0.43778265, 0.4630413, 0.44962096, 0.45289183, 0.36991775, 0.5137648, 0.42970267, 0.5123765, 0.46674812, 0.391997, 0.5428737, 0.26112723, 0.49296305, 0.4019059, 0.51857615, 0.34203234, 0.56297183, 0.4624285, 0.31473583, 0.48426515, 0.38961226, 0.4792106, 0.4556597, 0.3981469, 0.82181144, 0.076683015, 0.43709546, 0.45072502, 0.403063, 0.48123395, 0.445671, 0.43391934, 0.41085017, 0.47844657, 0.5523976, 0.3332042, 0.39573723, 0.4809303, 0.4767642, 0.3392432, 0.46270564, 0.42834285, 0.47794113, 0.48862976, 0.4476034, 0.39554918, 0.43229795, 0.39388365, 0.41972414, 0.44775766, 0.43749893, 0.44931862, 0.43717355, 0.43618527, 0.43013448, 0.4458758, 0.43667015, 0.4487376, 0.42987013, 0.45030433, 0.43525636, 0.43809214, 0.43754482, 0.44173053, 0.44406173, 0.43859297, 0.45381707, 0.43892053, 0.43997735, 0.44513983, 0.44053212, 0.44632536, 0.44307017, 0.43926865, 0.43738866, 0.4554001, 0.43955576, 0.43693265, 0.47264564, 0.42712784, 0.44716886, 0.44460776, 0.5509335, 0.36478138, 0.47435725, 0.4658819, 0.4416037, 0.5051691, 0.5688055, 0.31886065, 0.4494367, 0.46545348, 0.45204717, 0.48439282, 0.5093107, 0.43182653, 0.36819154, 0.40824592, 0.5469002, 0.44118574, 0.3783238, 0.46176466, 0.41548038, 0.40995488, 0.4822135, 0.5800308, 0.29410356, 0.46018916, 0.48044628, 0.41945794, 0.41041765, 0.48791614, 0.37282687, 0.50743467, 0.41529185, 0.47361425, 0.48168346, 0.4228861, 0.33253723, 0.5095234, 0.45906398, 0.642784, 0.23198482, 0.44205964, 0.4877115, 0.45557255, 0.4407478, 0.5385152, 0.322559, 0.48574784, 0.46824315, 0.39667577, 0.47068852, 0.39940956, 0.48200372, 0.4323175, 0.4799886, 0.44437137, 0.43002537, 0.48307553, 0.41584587, 0.59623086, 0.24069911, 0.56554425, 0.33221442, 0.4151645, 0.45958802, 0.43508443, 0.6425806, 0.21868126, 0.4656366, 0.4511151, 0.4845089, 0.4527534, 0.42543775, 0.4094777, 0.42367008, 0.5321767, 0.3677752, 0.47531018, 0.4517161, 0.37918308, 0.49726465, 0.38814765, 0.42835915, 0.4632879, 0.4440311, 0.47943822, 0.40242645, 0.43346727, 0.5168709, 0.39453724, 0.40909156, 0.6022595, 0.35694247, 0.47437656, 0.34844735, 0.41588652, 0.4761823, 0.4253355, 0.40495026, 0.48736376, 0.40208814, 0.4288367, 0.47124508, 0.4390421, 0.42400455, 0.4628096, 0.39726862, 0.46290562, 0.43853822, 0.41107228, 0.44229454, 0.44179985, 0.46654567, 0.4572283, 0.4161569, 0.43434337, 0.48350912, 0.3993831, 0.4577705, 0.44254288, 0.41447523, 0.44865403, 0.43820557, 0.45100358, 0.4392756, 0.44349754, 0.4503647, 0.43092006, 0.44659063, 0.44819656, 0.43131682, 0.44169182, 0.44011003, 0.438269, 0.51396376, 0.38224894, 0.4390596, 0.43750486, 0.45454222, 0.4378493, 0.45443264, 0.44270533, 0.4474091, 0.44448757, 0.45682675, 0.4980206, 0.41421643, 0.4514112, 0.46326315, 0.46560386, 0.53197193, 0.43952397, 0.49564803, 0.46847343, 0.40624964, 0.4064923, 0.49483812, 0.4149457, 0.4499877, 0.625206, 0.21759892, 0.53935164, 0.40051246, 0.45162553, 0.41494974, 0.4659547, 0.40767714, 0.50689864, 0.41831547, 0.49607942, 0.55310225, 0.26827705, 0.4272651, 0.4927804, 0.37623775, 0.45924285, 0.47485211, 0.41173732, 0.45295346, 0.4314743, 0.41183832, 0.4611953, 0.46668327, 0.3967664, 0.44442773, 0.51268476, 0.36944696, 0.5012953, 0.4221972, 0.4862213, 0.4474183, 0.41200915, 0.4791166, 0.40024853, 0.41146743, 0.5011146, 0.48578447, 0.40845227, 0.48076132, 0.3696122, 0.5005262, 0.3842459, 0.4731316, 0.4243625, 0.46022147, 0.44841248, 0.46589252, 0.45899782, 0.3922764, 0.42031428, 0.44252327, 0.4870819, 0.4538723, 0.44835046, 0.36659822, 0.5150066, 0.3900518, 0.63474655, 0.32462302, 0.45358902, 0.40390307, 0.41260695, 0.43972668, 0.42861447, 0.45983192, 0.45246503, 0.4192731, 0.4425529, 0.54830927, 0.33572084, 0.41461113, 0.5120946, 0.38407436, 0.45671064, 0.42534965, 0.45110303, 0.4793095, 0.41677573, 0.37993893, 0.50962234, 0.40329647, 0.44624412, 0.5046941, 0.47786894, 0.39184108, 0.4153896};


struct Mat* inputs[TRAIN_LEN];
struct Mat* targets[TRAIN_LEN];

void init(){
    for (int i = 0; i < SEQ_LEN; i++){
        cacheList[i] = (struct cache*)malloc(sizeof(struct cache));
    }
    a = newmat(1, 4*H, 0);
    prod_h_Wh = newmat(1, 4*H, 0);
    next_c = newmat(1, H, 0);
    next_h = newmat(1, H, 0);


    ig = newmat(1, H, 0);

    tanh_next_c = newmat(1, H, 0);

    prev_c = zeros(1, H);
    x_cur = newmat(1,INPUT_DIM,0);
    forward = (struct forward*)malloc(sizeof(struct forward));


    d1 = newmat(1, H, 0);
    one_matrix_H = ones(1,H);
    dop = newmat(1, H, 0);
    dprev_c = newmat(1,H,0);
    dfp = newmat(1,H,0);
    dip = newmat(1,H,0);
    dgp = newmat(1,H,0);
    do_ = newmat(1,H,0);
    df = newmat(1,H,0);
    di = newmat(1,H,0);
    dg = newmat(1,H,0);
    da = newmat(1,4 * H,0);
    db_step = newmat(1,4 * H,0);
    WxT = newmat(4*H, INPUT_DIM,0);
    WhT = newmat(4 * H, H,0);
    dx = newmat(1,INPUT_DIM,0);
    dprev_h = newmat(1,H,0);
    xT = newmat(INPUT_DIM, 1,0);
    dWx_step = newmat(INPUT_DIM,4 * H,0);
    prev_hT = newmat(H, 1,0);
    dWh_step = newmat(H, 4 * H, 0);
    step_backward = (struct step_backward*)malloc(sizeof(struct step_backward));

    dWx = newmat(INPUT_DIM,4 * H,0);
    dWh = newmat(H, 4*H, 0);
    dh_prev = newmat(1,H,0);
    dc_prev = newmat(1,H,0);
    db = newmat(1,4*H,0);


    prediction = newmat(1, OUTPUT_DIM, 0);


    prev_h = newmat(1,H,0); 
    dh_states = newmat(1,H,0);
    dWhy = newmat(OUTPUT_DIM,H,0);
    dby = newmat(1,OUTPUT_DIM,0);
    WhyT = newmat(H, OUTPUT_DIM, 0);
    scores = newmat(1, OUTPUT_DIM, 0);
    dscores = newmat(1, OUTPUT_DIM, 0);
    dscores2 = newmat(1, OUTPUT_DIM, 0);
    temp = newmat(1, OUTPUT_DIM, 0);
    one_matrix_O = ones(1,OUTPUT_DIM);
    backward = (struct backward*)malloc(sizeof(struct backward));

}


void create_training_data(){
    for(int i = 0; i < TRAIN_LEN; i++){
      inputs[i] = newmat(SEQ_LEN, 1, 0);
      targets[i] = newmat(1, OUTPUT_DIM, time_series[i + SEQ_LEN]);
      for(int k = 0; k < SEQ_LEN; k++){
        set(inputs[i], k + 1, 1, time_series[i + k]);
      }
    }
}

/*initialisation des poids avec des valeurs aléatoires*/
void init_weights(){
    Wh = randm(H, 4 * H, -1 / sqrtf(4 * H), 1 / sqrtf(4 * H));
    Wx = randm(INPUT_DIM, 4 * H, -1 / sqrtf(4 * H), 1 / sqrtf(4 * H));
    b = newmat(1, 4 * H,0);
    Why = randm(OUTPUT_DIM, H, -1 / sqrtf(H), 1 / sqrtf(H));
    by = newmat(1, OUTPUT_DIM, 0);
}





void app_main(void)
{
    printf("Hello world!\n");
    printf("sequence lenght : %d\n", SEQ_LEN);
    printf("hidden dimension : %d\n", H);
    init();
    init_weights();
    create_training_data();
    int nb_epochs = 100;
    clock_t begin = clock();

    float* loss_list = train(inputs, targets, INPUT_DIM, H, OUTPUT_DIM, TRAIN_LEN, SEQ_LEN, LEARNING_RATE, nb_epochs);
    printf("%s\n","loss :");
    float min_loss = loss_list[0];
    for(int i = 0; i<nb_epochs; i++){
        printf("%f\n",loss_list[i]);
        if (loss_list[i] < min_loss){
            min_loss = loss_list[i];
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("min loss : %f\n", min_loss);
    printf("Time spent : %f\n", time_spent);

	printf("Restarting now.\n");
	fflush(stdout);
	esp_restart();

}



